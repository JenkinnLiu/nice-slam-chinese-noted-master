import os
import time

import cv2
import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable

from src.common import (get_camera_from_tensor, get_samples,
                        get_tensor_from_camera, random_select)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer


class Mapper(object):
    """
    Mapper thread. Note that coarse mapper also uses this code.
    """

    def __init__(self, cfg, args, slam, coarse_mapper=False
                 ):

        self.cfg = cfg #NICE_SLAM.yaml
        self.args = args
        self.coarse_mapper = coarse_mapper #初始化时为false

        self.idx = slam.idx #当前帧id
        self.nice = slam.nice #使用niceslam
        self.c = slam.shared_c #共享c数组，及特征网格的值的形状
        self.bound = slam.bound #边界，(3*4),三行表示xyz，四列表示4层特征网格
        self.logger = slam.logger #初始化的logger
        self.mesher = slam.mesher #初始化的mesher
        self.output = slam.output
        self.verbose = slam.verbose #false
        self.renderer = slam.renderer
        self.low_gpu_mem = slam.low_gpu_mem #最低GPU显存
        self.mapping_idx = slam.mapping_idx #当前建图的帧id
        self.mapping_cnt = slam.mapping_cnt #已经有多少帧已经建好图了
        self.decoders = slam.shared_decoders #共享MLP
        self.estimate_c2w_list = slam.estimate_c2w_list #共享估计位姿
        self.mapping_first_frame = slam.mapping_first_frame #共享是否第一帧建图完成

        self.scale = cfg['scale'] #scale = 1
        self.coarse = cfg['coarse'] #coarse = True,肯定要粗糙建图
        self.occupancy = cfg['occupancy'] #体密度
        self.sync_method = cfg['sync_method'] #同步方法？sync_method = strict
        self.device = cfg['mapping']['device'] #cuda:0
        self.fix_fine = cfg['mapping']['fix_fine'] #True
        self.eval_rec = cfg['meshing']['eval_rec'] #false
        self.BA = False  # Even if BA is enabled, it starts only when there are at least 4 keyframes
        self.BA_cam_lr = cfg['mapping']['BA_cam_lr'] #0.001
        self.mesh_freq = cfg['mapping']['mesh_freq'] #50,特征网格更新频率？
        self.ckpt_freq = cfg['mapping']['ckpt_freq'] #500
        self.fix_color = cfg['mapping']['fix_color'] #false
        self.mapping_pixels = cfg['mapping']['pixels'] #建图体素：1000体素
        self.num_joint_iters = cfg['mapping']['iters'] #迭代次数：60
        self.clean_mesh = cfg['meshing']['clean_mesh'] #清空网格：true
        self.every_frame = cfg['mapping']['every_frame'] #5，每5帧建一次图
        self.color_refine = cfg['mapping']['color_refine'] #true，颜色细化
        self.w_color_loss = cfg['mapping']['w_color_loss'] #光度loss初始值0.2
        self.keyframe_every = cfg['mapping']['keyframe_every'] #每50帧选1个作为关键帧
        self.fine_iter_ratio = cfg['mapping']['fine_iter_ratio'] #fine层迭代率0.6
        self.middle_iter_ratio = cfg['mapping']['middle_iter_ratio'] #middle层迭代率0.4
        self.mesh_coarse_level = cfg['meshing']['mesh_coarse_level'] #false
        self.mapping_window_size = cfg['mapping']['mapping_window_size'] #建图时滑动窗口大小，5
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame'] #在第一帧不可视化true
        self.no_log_on_first_frame = cfg['mapping']['no_log_on_first_frame'] #在第一帧不log
        self.no_mesh_on_first_frame = cfg['mapping']['no_mesh_on_first_frame'] #在第一帧没有网格true
        self.frustum_feature_selection = cfg['mapping']['frustum_feature_selection'] #平截面特征选择 true
        self.keyframe_selection_method = cfg['mapping']['keyframe_selection_method'] #关键帧选择方法
        self.save_selected_keyframes_info = cfg['mapping']['save_selected_keyframes_info'] #保存选择的关键帧false
        if self.save_selected_keyframes_info:
            self.selected_keyframes = {} #存储所选择的关键帧

        if self.nice:
            if coarse_mapper:
                self.keyframe_selection_method = 'global'

        self.keyframe_dict = []
        self.keyframe_list = []
        self.frame_reader = get_dataset(
            cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        if 'Demo' not in self.output:  # disable this visualization in demo
            self.visualizer = Visualizer(freq=cfg['mapping']['vis_freq'], inside_freq=cfg['mapping']['vis_inside_freq'],
                                         vis_dir=os.path.join(self.output, 'mapping_vis'), renderer=self.renderer,
                                         verbose=self.verbose, device=self.device)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def get_mask_from_c2w(self, c2w, key, val_shape, depth_np):
        """
        Frustum feature selection based on current camera pose and depth image.
        根据当前相机位姿和深度图来选择平截面特征

        Args:
            c2w (tensor): camera pose of current frame.
            key (str): name of this feature grid.
            val_shape (tensor): shape of the grid.
            depth_np (numpy.array): depth image of current frame.

        Returns:
            mask (tensor): mask for selected optimizable feature.
            points (tensor): corresponding point coordinates.
        """
        H, W, fx, fy, cx, cy, = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        X, Y, Z = torch.meshgrid(torch.linspace(self.bound[0][0], self.bound[0][1], val_shape[2]),
                                 torch.linspace(self.bound[1][0], self.bound[1][1], val_shape[1]),
                                 torch.linspace(self.bound[2][0], self.bound[2][1], val_shape[0]))

        points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
        if key == 'grid_coarse':
            mask = np.ones(val_shape[::-1]).astype(np.bool_)
            return mask
        points_bak = points.clone()
        c2w = c2w.cpu().numpy()
        w2c = np.linalg.inv(c2w)
        ones = np.ones_like(points[:, 0]).reshape(-1, 1)
        homo_vertices = np.concatenate(
            [points, ones], axis=1).reshape(-1, 4, 1)
        cam_cord_homo = w2c @ homo_vertices
        cam_cord = cam_cord_homo[:, :3]
        K = np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)
        cam_cord[:, 0] *= -1
        uv = K @ cam_cord
        z = uv[:, -1:] + 1e-5
        uv = uv[:, :2] / z
        uv = uv.astype(np.float32)

        remap_chunk = int(3e4)
        depths = []
        for i in range(0, uv.shape[0], remap_chunk):
            depths += [cv2.remap(depth_np,
                                 uv[i:i + remap_chunk, 0],
                                 uv[i:i + remap_chunk, 1],
                                 interpolation=cv2.INTER_LINEAR)[:, 0].reshape(-1, 1)]
        depths = np.concatenate(depths, axis=0)

        edge = 0
        mask = (uv[:, 0] < W - edge) * (uv[:, 0] > edge) * \
               (uv[:, 1] < H - edge) * (uv[:, 1] > edge)

        # For ray with depth==0, fill it with maximum depth
        zero_mask = (depths == 0)
        depths[zero_mask] = np.max(depths)

        # depth test
        mask = mask & (0 <= -z[:, :, 0]) & (-z[:, :, 0] <= depths + 0.5)
        mask = mask.reshape(-1)

        # add feature grid near cam center
        ray_o = c2w[:3, 3]
        ray_o = torch.from_numpy(ray_o).unsqueeze(0)

        dist = points_bak - ray_o
        dist = torch.sum(dist * dist, axis=1)
        mask2 = dist < 0.5 * 0.5
        mask2 = mask2.cpu().numpy()
        mask = mask | mask2

        points = points[mask]
        mask = mask.reshape(val_shape[2], val_shape[1], val_shape[0])
        return mask

    def keyframe_selection_overlap(self, gt_color, gt_depth, c2w, keyframe_dict, k, N_samples=16, pixels=100):
        """
        Select overlapping keyframes to the current camera observation.
        选择当前摄影机观察的重叠关键帧
        Args:
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            c2w (tensor): camera to world matrix (3*4 or 4*4 both fine).
            keyframe_dict (list): a list containing info for each keyframe.
            k (int): number of overlapping keyframes to select.
            N_samples (int, optional): number of samples/points per ray. Defaults to 16.
            pixels (int, optional): number of pixels to sparsely sample 
                from the image of the current camera. Defaults to 100.
        Returns:
            selected_keyframe_list (list): list of selected keyframe id.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        rays_o, rays_d, gt_depth, gt_color = get_samples(
            0, H, 0, W, pixels, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)

        gt_depth = gt_depth.reshape(-1, 1)
        gt_depth = gt_depth.repeat(1, N_samples)
        t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
        near = gt_depth * 0.8
        far = gt_depth + 0.5
        z_vals = near * (1. - t_vals) + far * (t_vals)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
              z_vals[..., :, None]  # [N_rays, N_samples, 3]
        vertices = pts.reshape(-1, 3).cpu().numpy()
        list_keyframe = []
        for keyframeid, keyframe in enumerate(keyframe_dict):
            c2w = keyframe['est_c2w'].cpu().numpy()
            w2c = np.linalg.inv(c2w)
            ones = np.ones_like(vertices[:, 0]).reshape(-1, 1)
            homo_vertices = np.concatenate(
                [vertices, ones], axis=1).reshape(-1, 4, 1)  # (N, 4)
            cam_cord_homo = w2c @ homo_vertices  # (N, 4, 1)=(4,4)*(N, 4, 1)
            cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)
            K = np.array([[fx, .0, cx], [.0, fy, cy],
                          [.0, .0, 1.0]]).reshape(3, 3)
            cam_cord[:, 0] *= -1
            uv = K @ cam_cord
            z = uv[:, -1:] + 1e-5
            uv = uv[:, :2] / z
            uv = uv.astype(np.float32)
            edge = 20
            mask = (uv[:, 0] < W - edge) * (uv[:, 0] > edge) * \
                   (uv[:, 1] < H - edge) * (uv[:, 1] > edge)
            mask = mask & (z[:, :, 0] < 0)
            mask = mask.reshape(-1)
            percent_inside = mask.sum() / uv.shape[0]
            list_keyframe.append(
                {'id': keyframeid, 'percent_inside': percent_inside})

        list_keyframe = sorted(
            list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
        selected_keyframe_list = [dic['id']
                                  for dic in list_keyframe if dic['percent_inside'] > 0.00]
        #选择dic['percent_inside'] > 0.0 的关键帧
        selected_keyframe_list = list(np.random.permutation(
            np.array(selected_keyframe_list))[:k])
        return selected_keyframe_list

    def optimize_map(self, num_joint_iters, lr_factor, idx, cur_gt_color, cur_gt_depth, gt_cur_c2w, keyframe_dict,
                     keyframe_list, cur_c2w):
        """
        Mapping iterations. Sample pixels from selected keyframes,
        then optimize scene representation and camera poses(if local BA enabled).
        从选择的关键帧里采样，并优化场景表征和相机位姿（局部BA）
        Args:
            num_joint_iters (int): number of mapping iterations.
            lr_factor (float): the factor to times on current lr.
            idx (int): the index of current frame
            cur_gt_color (tensor): gt_color image of the current camera.
            cur_gt_depth (tensor): gt_depth image of the current camera.
            gt_cur_c2w (tensor): groundtruth camera to world matrix corresponding to current frame.
            keyframe_dict (list): list of keyframes info dictionary.
            keyframe_list (list): list ofkeyframe index.
            cur_c2w (tensor): the estimated camera to world matrix of current frame. 

        Returns:
            cur_c2w/None (tensor/None): return the updated cur_c2w, return None if no BA
        """
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy # 相机内参
        c = self.c
        cfg = self.cfg
        device = self.device
        bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
            [1, 4])).type(torch.float32).to(device)

        if len(keyframe_dict) == 0:
            optimize_frame = []
        else:
            if self.keyframe_selection_method == 'global':
                num = self.mapping_window_size - 2
                optimize_frame = random_select(len(self.keyframe_dict) - 1, num)
            elif self.keyframe_selection_method == 'overlap':  #关键帧选择方法为overlap（初始化时默认选overlap）
                num = self.mapping_window_size - 2 #滑动窗口大小为2
                optimize_frame = self.keyframe_selection_overlap(   #选择待优化的关键帧
                    cur_gt_color, cur_gt_depth, cur_c2w, keyframe_dict[:-1], num)

        # add the last keyframe and the current frame(use -1 to denote)
        oldest_frame = None
        if len(keyframe_list) > 0:   #如果有关键帧可以优化
            optimize_frame = optimize_frame + [len(keyframe_list) - 1]  #optimize_frame是一个list
            oldest_frame = min(optimize_frame)
        optimize_frame += [-1] #插入当前帧作为待优化帧

        if self.save_selected_keyframes_info:
            keyframes_info = []
            for id, frame in enumerate(optimize_frame):
                if frame != -1:  #如果不是当前帧
                    frame_idx = keyframe_list[frame]  #就从groundtruth里找
                    tmp_gt_c2w = keyframe_dict[frame]['gt_c2w']
                    tmp_est_c2w = keyframe_dict[frame]['est_c2w']
                else: #如果是当前帧
                    frame_idx = idx
                    tmp_gt_c2w = gt_cur_c2w
                    tmp_est_c2w = cur_c2w
                keyframes_info.append(
                    {'idx': frame_idx, 'gt_c2w': tmp_gt_c2w, 'est_c2w': tmp_est_c2w})
            self.selected_keyframes[idx] = keyframes_info

        pixs_per_image = self.mapping_pixels // len(optimize_frame)
        # 准备好了帧id，真实位姿和估计位姿这三个信息，准备放入decoder
        decoders_para_list = [] #decoder参数
        coarse_grid_para = [] #粗糙级网格参数
        middle_grid_para = [] #中等级网格参数
        fine_grid_para = [] #精细级网格参数
        color_grid_para = [] #颜色网格参数
        gt_depth_np = cur_gt_depth.cpu().numpy()
        if self.nice:
            if self.frustum_feature_selection:  #选择平截面特征
                masked_c_grad = {}
                mask_c2w = cur_c2w  #初始化mask的c2w
            for key, val in c.items():
                if not self.frustum_feature_selection:  #如果没有平截面特征，直接放进网格里
                    val = Variable(val.to(device), requires_grad=True) #计算出网格的值
                    c[key] = val  #放进不同层级的网格里
                    if key == 'grid_coarse':
                        coarse_grid_para.append(val)
                    elif key == 'grid_middle':
                        middle_grid_para.append(val)
                    elif key == 'grid_fine':
                        fine_grid_para.append(val)
                    elif key == 'grid_color':
                        color_grid_para.append(val)

                else: #如果有平截面特征
                    mask = self.get_mask_from_c2w(  #根据当前相机位姿和深度图来选择平截面特征
                        mask_c2w, key, val.shape[2:], gt_depth_np)
                    mask = torch.from_numpy(mask).permute(2, 1, 0).unsqueeze(
                        0).unsqueeze(0).repeat(1, val.shape[1], 1, 1, 1)  #转换一下维度，把第一维和第三维交换
                    val = val.to(device)
                    # val_grad is the optimizable part, other parameters will be fixed
                    val_grad = val[mask].clone()
                    val_grad = Variable(val_grad.to(
                        device), requires_grad=True)
                    masked_c_grad[key] = val_grad
                    masked_c_grad[key + 'mask'] = mask #连带平截面特征一起放进不同层级的网格里
                    if key == 'grid_coarse':
                        coarse_grid_para.append(val_grad)
                    elif key == 'grid_middle':
                        middle_grid_para.append(val_grad)
                    elif key == 'grid_fine':
                        fine_grid_para.append(val_grad)
                    elif key == 'grid_color':
                        color_grid_para.append(val_grad)

        if self.nice:
            if not self.fix_fine:  #如果不固定住fine层
                decoders_para_list += list(
                    self.decoders.fine_decoder.parameters())
            if not self.fix_color:
                decoders_para_list += list(
                    self.decoders.color_decoder.parameters())
        else:
            # imap*, single MLP
            decoders_para_list += list(self.decoders.parameters())

        if self.BA:
            camera_tensor_list = []
            gt_camera_tensor_list = []
            for frame in optimize_frame:
                # the oldest frame should be fixed to avoid drifting
                if frame != oldest_frame: #不带旧的帧
                    if frame != -1:  #如果不是当前帧
                        c2w = keyframe_dict[frame]['est_c2w']   #查询估计位姿和真实位姿
                        gt_c2w = keyframe_dict[frame]['gt_c2w']
                    else:
                        c2w = cur_c2w
                        gt_c2w = gt_cur_c2w
                    camera_tensor = get_tensor_from_camera(c2w)
                    camera_tensor = Variable(
                        camera_tensor.to(device), requires_grad=True)
                    camera_tensor_list.append(camera_tensor) #相机位姿的list
                    gt_camera_tensor = get_tensor_from_camera(gt_c2w)
                    gt_camera_tensor_list.append(gt_camera_tensor)

        if self.nice:
            if self.BA:  #BA只优化相机位姿，其余会更新地图
                # The corresponding lr will be set according to which stage the optimization is in
                optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                              {'params': coarse_grid_para, 'lr': 0},
                                              {'params': middle_grid_para, 'lr': 0},
                                              {'params': fine_grid_para, 'lr': 0},
                                              {'params': color_grid_para, 'lr': 0},
                                              {'params': camera_tensor_list, 'lr': 0}])
            else:
                optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                              {'params': coarse_grid_para, 'lr': 0},
                                              {'params': middle_grid_para, 'lr': 0},
                                              {'params': fine_grid_para, 'lr': 0},
                                              {'params': color_grid_para, 'lr': 0}])
        else:
            # imap*, single MLP
            if self.BA:
                optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                              {'params': camera_tensor_list, 'lr': 0}])
            else:
                optimizer = torch.optim.Adam(
                    [{'params': decoders_para_list, 'lr': 0}])
            from torch.optim.lr_scheduler import StepLR
            scheduler = StepLR(optimizer, step_size=200, gamma=0.8)

        for joint_iter in range(num_joint_iters): #开始迭代训练
            if self.nice:
                if self.frustum_feature_selection:  #如果有平截面特征
                    for key, val in c.items():
                        if (self.coarse_mapper and 'coarse' in key) or \
                                ((not self.coarse_mapper) and ('coarse' not in key)):
                            val_grad = masked_c_grad[key] #优化具有平截面特征的网格
                            mask = masked_c_grad[key + 'mask'] #得到平截面
                            val = val.to(device)
                            val[mask] = val_grad
                            c[key] = val #把带mask的值赋值给c

                if self.coarse_mapper:
                    self.stage = 'coarse'
                elif joint_iter <= int(num_joint_iters * self.middle_iter_ratio):
                    self.stage = 'middle'
                elif joint_iter <= int(num_joint_iters * self.fine_iter_ratio):
                    self.stage = 'fine'
                else:
                    self.stage = 'color'
                #从cfg加载各层的lr
                optimizer.param_groups[0]['lr'] = cfg['mapping']['stage'][self.stage]['decoders_lr'] * lr_factor
                optimizer.param_groups[1]['lr'] = cfg['mapping']['stage'][self.stage]['coarse_lr'] * lr_factor
                optimizer.param_groups[2]['lr'] = cfg['mapping']['stage'][self.stage]['middle_lr'] * lr_factor
                optimizer.param_groups[3]['lr'] = cfg['mapping']['stage'][self.stage]['fine_lr'] * lr_factor
                optimizer.param_groups[4]['lr'] = cfg['mapping']['stage'][self.stage]['color_lr'] * lr_factor
                if self.BA:
                    if self.stage == 'color':
                        optimizer.param_groups[5]['lr'] = self.BA_cam_lr  #优化相机位姿的lr
            else:
                self.stage = 'color'
                optimizer.param_groups[0]['lr'] = cfg['mapping']['imap_decoders_lr']
                if self.BA:
                    optimizer.param_groups[1]['lr'] = self.BA_cam_lr

            if (not (idx == 0 and self.no_vis_on_first_frame)) and ('Demo' not in self.output):
                self.visualizer.vis(
                    idx, joint_iter, cur_gt_depth, cur_gt_color, cur_c2w, self.c, self.decoders)

            optimizer.zero_grad() # 清空梯度
            batch_rays_d_list = []
            batch_rays_o_list = []
            batch_gt_depth_list = []
            batch_gt_color_list = []

            camera_tensor_id = 0
            for frame in optimize_frame: #遍历所有待优化帧
                if frame != -1: #如果不是当前帧
                    gt_depth = keyframe_dict[frame]['depth'].to(device)
                    gt_color = keyframe_dict[frame]['color'].to(device)
                    if self.BA and frame != oldest_frame:
                        camera_tensor = camera_tensor_list[camera_tensor_id]
                        camera_tensor_id += 1
                        c2w = get_camera_from_tensor(camera_tensor)
                    else:
                        c2w = keyframe_dict[frame]['est_c2w']

                else: #如果是当前帧
                    gt_depth = cur_gt_depth.to(device) #ground_truth RGBD
                    gt_color = cur_gt_color.to(device)
                    if self.BA:
                        camera_tensor = camera_tensor_list[camera_tensor_id]
                        c2w = get_camera_from_tensor(camera_tensor) #得到相机位姿
                    else:
                        c2w = cur_c2w

                batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(  #根据位姿计算采样射线，ray_o为射线中心，ray_d为射线方向
                    0, H, 0, W, pixs_per_image, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)
                batch_rays_o_list.append(batch_rays_o.float()) #将射线等参数装入batch的list里
                batch_rays_d_list.append(batch_rays_d.float())
                batch_gt_depth_list.append(batch_gt_depth.float())
                batch_gt_color_list.append(batch_gt_color.float())

            batch_rays_d = torch.cat(batch_rays_d_list)
            batch_rays_o = torch.cat(batch_rays_o_list)
            batch_gt_depth = torch.cat(batch_gt_depth_list)
            batch_gt_color = torch.cat(batch_gt_color_list)

            if self.nice:
                # should pre-filter those out of bounding box depth value
                # 应该提前过滤掉那些在边界框之外的深度值
                with torch.no_grad():
                    det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                    det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                    t = (self.bound.unsqueeze(0).to(
                        device) - det_rays_o) / det_rays_d
                    t, _ = torch.min(torch.max(t, dim=2)[0], dim=1) #过滤掉边界框之外的深度值
                    inside_mask = t >= batch_gt_depth
                batch_rays_d = batch_rays_d[inside_mask]
                batch_rays_o = batch_rays_o[inside_mask]
                batch_gt_depth = batch_gt_depth[inside_mask]
                batch_gt_color = batch_gt_color[inside_mask]
                #通过采样射线rays_o和rays_d和深度图即可渲染
            ret = self.renderer.render_batch_ray(c, self.decoders, batch_rays_d,  #开始放入神经网络，渲染一下
                                                 batch_rays_o, device, self.stage,
                                                 gt_depth=None if self.coarse_mapper else batch_gt_depth)
            depth, uncertainty, color = ret  #得到了渲染之后的深度、颜色、和不确定性

            depth_mask = (batch_gt_depth > 0)
            loss = torch.abs(
                batch_gt_depth[depth_mask] - depth[depth_mask]).sum()  #计算深度loss，即几何损失loss
            if ((not self.nice) or (self.stage == 'color')):
                color_loss = torch.abs(batch_gt_color - color).sum()
                weighted_color_loss = self.w_color_loss * color_loss
                loss += weighted_color_loss  #计算光度损失

            # for imap*, it uses volume density
            regulation = (not self.occupancy)
            if regulation:
                point_sigma = self.renderer.regulation(
                    c, self.decoders, batch_rays_d, batch_rays_o, batch_gt_depth, device, self.stage)
                regulation_loss = torch.abs(point_sigma).sum()
                loss += 0.0005 * regulation_loss

            loss.backward(retain_graph=False) #距离optimizer.zero_grad()比较远
            optimizer.step()
            if not self.nice:
                # for imap*
                scheduler.step()
            optimizer.zero_grad()

            # put selected and updated features back to the grid
            if self.nice and self.frustum_feature_selection:  #特征优化完毕，下一步就是要把优化后的特征塞到特征网格里
                for key, val in c.items():
                    if (self.coarse_mapper and 'coarse' in key) or \
                            ((not self.coarse_mapper) and ('coarse' not in key)):
                        val_grad = masked_c_grad[key]
                        mask = masked_c_grad[key + 'mask']
                        val = val.detach()
                        val[mask] = val_grad.clone().detach()
                        c[key] = val

        if self.BA:
            # put the updated camera poses back
            # 把优化后的相机位姿塞回去
            camera_tensor_id = 0
            for id, frame in enumerate(optimize_frame):
                if frame != -1:
                    if frame != oldest_frame:
                        c2w = get_camera_from_tensor(
                            camera_tensor_list[camera_tensor_id].detach())
                        c2w = torch.cat([c2w, bottom], dim=0)
                        camera_tensor_id += 1
                        keyframe_dict[frame]['est_c2w'] = c2w.clone()
                else:
                    c2w = get_camera_from_tensor(
                        camera_tensor_list[-1].detach())
                    c2w = torch.cat([c2w, bottom], dim=0)
                    cur_c2w = c2w.clone()
        if self.BA:
            return cur_c2w
        else:
            return None

    def run(self):
        cfg = self.cfg
        idx, gt_color, gt_depth, gt_c2w = self.frame_reader[0] #dataLoader,

        self.estimate_c2w_list[0] = gt_c2w.cpu() #把初始位姿读入内存，存到初始位姿里
        init = True #初始化完成
        prev_idx = -1
        while (1):
            while True:
                idx = self.idx[0].clone() #当前帧id
                if idx == self.n_img - 1:   #如果当前帧为最后一帧，则继续
                    break
                if self.sync_method == 'strict':  #同步模式为strict（默认模式）
                    if idx % self.every_frame == 0 and idx != prev_idx:   #如果idx到了建图的时候且当前帧没建图完成，则继续
                        break

                elif self.sync_method == 'loose':
                    if idx == 0 or idx >= prev_idx + self.every_frame // 2:
                        break
                elif self.sync_method == 'free':
                    break
                time.sleep(0.1)
            prev_idx = idx

            if self.verbose:
                print(Fore.GREEN)
                prefix = 'Coarse ' if self.coarse_mapper else ''
                print(prefix + "Mapping Frame ", idx.item())
                print(Style.RESET_ALL)

            _, gt_color, gt_depth, gt_c2w = self.frame_reader[idx] #当前帧的RGB-D和真实位姿

            if not init: #代表建图完成，但追踪线程一直没有传递关键帧消息，这里就再迭代个比较少的次数来细化颜色，正常是要跑1500次的
                lr_factor = cfg['mapping']['lr_factor']
                num_joint_iters = cfg['mapping']['iters']  #60

                # here provides a color refinement postprocess 颜色细化过程
                #如果说当前帧为最后一帧，加载完成，跟踪线程没有任何消息
                if idx == self.n_img - 1 and self.color_refine and not self.coarse_mapper:
                    outer_joint_iters = 5 #设outer_joint_iters迭代次数为5次
                    self.mapping_window_size *= 2 #滑动窗口扩大两倍
                    self.middle_iter_ratio = 0.0
                    self.fine_iter_ratio = 0.0  #暂停middle和fine
                    num_joint_iters *= 5  #精细化迭代次数*=5
                    self.fix_color = True
                    self.frustum_feature_selection = False
                else:
                    if self.nice:
                        outer_joint_iters = 1
                    else:
                        outer_joint_iters = 3

            else: #如果已经初始化完成了，迭代1500次
                outer_joint_iters = 1
                lr_factor = cfg['mapping']['lr_first_factor']
                num_joint_iters = cfg['mapping']['iters_first'] #1500次

            cur_c2w = self.estimate_c2w_list[idx].to(self.device) #位姿
            num_joint_iters = num_joint_iters // outer_joint_iters  #迭代1500次
            for outer_joint_iter in range(outer_joint_iters):

                self.BA = (len(self.keyframe_list) > 4) and cfg['mapping']['BA'] and ( #BA需要4个关键帧
                    not self.coarse_mapper) #而且不是粗糙级的图

                _ = self.optimize_map(num_joint_iters, lr_factor, idx, gt_color, gt_depth, #更新下地图和位姿
                                      gt_c2w, self.keyframe_dict, self.keyframe_list, cur_c2w=cur_c2w)
                if self.BA:
                    cur_c2w = _
                    self.estimate_c2w_list[idx] = cur_c2w  #优化后的位姿

                # add new frame to keyframe set
                if outer_joint_iter == outer_joint_iters - 1:
                    if (idx % self.keyframe_every == 0 or (idx == self.n_img - 2)) \
                            and (idx not in self.keyframe_list):
                        self.keyframe_list.append(idx)
                        self.keyframe_dict.append({'gt_c2w': gt_c2w.cpu(), 'idx': idx, 'color': gt_color.cpu(
                        ), 'depth': gt_depth.cpu(), 'est_c2w': cur_c2w.clone()})

            if self.low_gpu_mem:
                torch.cuda.empty_cache() #节省显存

            init = False #第一次建图完成，可以追踪了，这里把init设为false等待下一次建图
            #下次如果init再为True代表追踪完成之后开启建图线程，跑1500次，如果init一直为false就代表一直在建图
            # mapping of first frame is done, can begin tracking
            self.mapping_first_frame[0] = 1

            if not self.coarse_mapper: #如果没有粗糙建图模式
                if ((not (idx == 0 and self.no_log_on_first_frame)) and idx % self.ckpt_freq == 0) \
                        or idx == self.n_img - 1:
                    self.logger.log(idx, self.keyframe_dict, self.keyframe_list,  #将参数保存到文件，即建立log日志
                                    selected_keyframes=self.selected_keyframes
                                    if self.save_selected_keyframes_info else None)

                self.mapping_idx[0] = idx
                self.mapping_cnt[0] += 1 #更新正在建图的帧

                if (idx % self.mesh_freq == 0) and (not (idx == 0 and self.no_mesh_on_first_frame)): #如果到了建立mesh的时候了
                    mesh_out_file = f'{self.output}/mesh/{idx:05d}_mesh.ply'
                    self.mesher.get_mesh(mesh_out_file, self.c, self.decoders, self.keyframe_dict, #建立mesh
                                         self.estimate_c2w_list,
                                         idx, self.device, show_forecast=self.mesh_coarse_level,
                                         clean_mesh=self.clean_mesh, get_mask_use_all_frames=False)

                if idx == self.n_img - 1:
                    mesh_out_file = f'{self.output}/mesh/final_mesh.ply'
                    self.mesher.get_mesh(mesh_out_file, self.c, self.decoders, self.keyframe_dict,
                                         self.estimate_c2w_list,
                                         idx, self.device, show_forecast=self.mesh_coarse_level,
                                         clean_mesh=self.clean_mesh, get_mask_use_all_frames=False)
                    os.system(
                        f"cp {mesh_out_file} {self.output}/mesh/{idx:05d}_mesh.ply")
                    if self.eval_rec:
                        mesh_out_file = f'{self.output}/mesh/final_mesh_eval_rec.ply'
                        self.mesher.get_mesh(mesh_out_file, self.c, self.decoders, self.keyframe_dict,
                                             self.estimate_c2w_list, idx, self.device, show_forecast=False,
                                             clean_mesh=self.clean_mesh, get_mask_use_all_frames=True)
                    break

            if idx == self.n_img - 1:
                break
