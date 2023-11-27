import copy
import os
import time

import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import (get_camera_from_tensor, get_samples,
                        get_tensor_from_camera)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer


class Tracker(object):
    def __init__(self, cfg, args, slam
                 ):
        self.cfg = cfg  # 未使用
        self.args = args  # 未使用

        self.scale = cfg['scale'] #1
        self.coarse = cfg['coarse']  # 未使用
        self.occupancy = cfg['occupancy']  # 未使用
        self.sync_method = cfg['sync_method'] #同步方式strict

        self.idx = slam.idx #当前帧id
        self.nice = slam.nice #使用nice-slam
        self.bound = slam.bound #边界
        self.mesher = slam.mesher  # 初始化的mesher未使用
        self.output = slam.output
        self.verbose = slam.verbose #false
        self.shared_c = slam.shared_c #共享特征网格
        self.renderer = slam.renderer
        self.gt_c2w_list = slam.gt_c2w_list #存储真实位姿的list
        self.low_gpu_mem = slam.low_gpu_mem #低显存模式
        self.mapping_idx = slam.mapping_idx #正在建图的帧id
        self.mapping_cnt = slam.mapping_cnt  #统计已经有多少帧建过图
        self.shared_decoders = slam.shared_decoders #共享decoder
        self.estimate_c2w_list = slam.estimate_c2w_list #存储估计位姿的list

        self.cam_lr = cfg['tracking']['lr'] #0.001,1e-3
        self.device = cfg['tracking']['device'] #"cuda:0"
        self.num_cam_iters = cfg['tracking']['iters'] #相机跟踪迭代次数？10
        self.gt_camera = cfg['tracking']['gt_camera'] #false，感觉是用来更新位姿的
        self.tracking_pixels = cfg['tracking']['pixels'] #200
        self.seperate_LR = cfg['tracking']['seperate_LR'] #false
        self.w_color_loss = cfg['tracking']['w_color_loss'] #weighted_color_loss光度损失（颜色损失），0.5
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W'] #忽略W边：20，这两是干啥用的？
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H'] #忽略H边：20
        self.handle_dynamic = cfg['tracking']['handle_dynamic'] #True:对动态目标具有鲁棒性，忽略动态物体
        self.use_color_in_tracking = cfg['tracking']['use_color_in_tracking'] #True，跟踪时使用颜色
        self.const_speed_assumption = cfg['tracking']['const_speed_assumption'] #True，速度快时使用该估计策略？

        self.every_frame = cfg['mapping']['every_frame'] #5，每5帧来一次建图？
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame'] #True建图时不显示第一帧

        self.prev_mapping_idx = -1  #之前用于建图的帧，初始化为-1
        self.frame_reader = get_dataset(
            cfg, args, self.scale, device=self.device) #DataLoader
        self.n_img = len(self.frame_reader) #图片的数量
        self.frame_loader = DataLoader(
            self.frame_reader, batch_size=1, shuffle=False, num_workers=1)
        self.visualizer = Visualizer(freq=cfg['tracking']['vis_freq'], inside_freq=cfg['tracking']['vis_inside_freq'],
                                     vis_dir=os.path.join(self.output,
                                                          'vis' if 'Demo' in self.output else 'tracking_vis'),
                                     renderer=self.renderer, verbose=self.verbose, device=self.device)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def optimize_cam_in_batch(self, camera_tensor, gt_color, gt_depth, batch_size, optimizer):
        """
        Do one iteration of camera iteration. Sample pixels, render depth/color, calculate loss and backpropagation.
        迭代一次相机的跟踪操作，对像素点进行采样，渲染深度和颜色，计算两个loss并反向传播，以此进行优化，一次处理batch_size个数据
        Args:
            camera_tensor (tensor): camera tensor.
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            batch_size (int): batch size, number of sampling rays.
            optimizer (torch.optim): camera optimizer.

        Returns:
            loss (float): The value of loss.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        optimizer.zero_grad() #清空梯度
        c2w = get_camera_from_tensor(camera_tensor) #将变换矩阵RT变成四元数quad和平移T
        Wedge = self.ignore_edge_W #忽略W边：20
        Hedge = self.ignore_edge_H #忽略H边：20 ，这两干啥用的？？？
        # 下面根据位姿计算采样射线，ray_o为射线中心，ray_d为射线方向
        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples( #从图像区域H0~H1、W0~W1获取n条射线,并用射线进行采样
            Hedge, H - Hedge, Wedge, W - Wedge, batch_size, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)
        if self.nice:
            # should pre-filter those out of bounding box depth value
            # 应该提前过滤掉那些边缘之外的值
            with torch.no_grad():
                det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)，unsqueeze(-1)表示在末尾加一维
                det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                t = (self.bound.unsqueeze(0).to(device) - det_rays_o) / det_rays_d
                t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                inside_mask = t >= batch_gt_depth #这里没太懂，就当是一种数据预处理吧，过滤掉那些边缘之外的深度值
            batch_rays_d = batch_rays_d[inside_mask]
            batch_rays_o = batch_rays_o[inside_mask]
            batch_gt_depth = batch_gt_depth[inside_mask]
            batch_gt_color = batch_gt_color[inside_mask]
        # 通过采样射线rays_o和rays_d和深度图即可渲染
        ret = self.renderer.render_batch_ray( #开始放入神经网络，渲染一下
            self.c, self.decoders, batch_rays_d, batch_rays_o, self.device, stage='color', gt_depth=batch_gt_depth)
        depth, uncertainty, color = ret   #得到了渲染之后的深度、深度方差，和颜色

        uncertainty = uncertainty.detach() #深度方差
        if self.handle_dynamic: #把动态物体处理掉，怎么处理呢？
            #如果有动态物体，则batch_gt_depth - depth的值会比较大，把它mask掉，即mask取0。其中uncertainty是用于处理动态物体的
            tmp = torch.abs(batch_gt_depth - depth) / torch.sqrt(uncertainty + 1e-10) #tmp大于其中位数的10倍，代表检测出动态物体
            #.median是中位数，如果检测到动态物体，mask数组的下标为0
            mask = (tmp < 10 * tmp.median()) & (batch_gt_depth > 0) #mask是下标，形式为[1,0,1,0]，可以不取用哪些下标来计算误差
        else:
            mask = batch_gt_depth > 0

        loss = (torch.abs(batch_gt_depth - depth) /
                torch.sqrt(uncertainty + 1e-10))[mask].sum() #论文中改进的几何loss，uncertainty为深度方差

        if self.use_color_in_tracking:
            color_loss = torch.abs(
                batch_gt_color - color)[mask].sum() #计算mask掉动态物体的颜色损失loss
            loss += self.w_color_loss * color_loss #颜色损失是与几何损失直接相加的

        loss.backward() #误差反向传播
        optimizer.step() #更新参数
        optimizer.zero_grad() #清空梯度

        return loss.item() #返回loss

    def update_para_from_mapping(self):
        """
        Update the parameters of scene representation from the mapping thread.
        从mapping线程更新场景表征的参数（看起来像是在建图过后就要调用一下这个），更新本地c数组
        """
        if self.mapping_idx[0] != self.prev_mapping_idx:
            if self.verbose:
                print('Tracking: update the parameters from mapping')
            self.decoders = copy.deepcopy(self.shared_decoders).to(self.device)
            for key, val in self.shared_c.items(): #tracking线程从mapping线程的共享特征网格shared_c数组中提取key和值
                val = val.clone().to(self.device)
                self.c[key] = val #更新本地c数组
            self.prev_mapping_idx = self.mapping_idx[0].clone() #更新上一次建图的帧

    def run(self):
        device = self.device
        self.c = {}  #初始化c数组
        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader)

        for idx, gt_color, gt_depth, gt_c2w in pbar:
            if not self.verbose: #加载当前帧的颜色、深度、位姿
                pbar.set_description(f"Tracking Frame {idx[0]}")

            idx = idx[0] #帧id
            gt_depth = gt_depth[0] #颜色
            gt_color = gt_color[0] #深度
            gt_c2w = gt_c2w[0] #位姿
            if self.sync_method == 'strict': #strict模式
                # strictly mapping and then tracking
                # 严格按照先建图再追踪再建图再追踪的模式
                # initiate mapping every self.every_frame frames
                # 每过every_frame帧，则建一次图
                if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1): #到了建图的时候了
                    while self.mapping_idx[0] != idx - 1: #那就建一次图吧
                        time.sleep(0.1) #Track线程要休息了，让给建图线程了
                    pre_c2w = self.estimate_c2w_list[idx - 1].to(device) #保存上一次的估计位姿
            elif self.sync_method == 'loose':
                # loose模式，建图可能在追踪之后了
                # mapping idx can be later than tracking idx is within the bound of
                # [-self.every_frame-self.every_frame//2, -self.every_frame+self.every_frame//2]
                while self.mapping_idx[0] < idx - self.every_frame - self.every_frame // 2:
                    time.sleep(0.1)
            elif self.sync_method == 'free': #单纯的并行
                # pure parallel, if mesh/vis happens may cause inbalance
                pass

            self.update_para_from_mapping()# 从mapping线程更新场景表征的参数，更新本地c数组（看起来像是在建图过后就要调用一下这个）

            if self.verbose:
                print(Fore.MAGENTA) #酷炫字体着色效果
                print("Tracking Frame ", idx.item())
                print(Style.RESET_ALL)

            if idx == 0 or self.gt_camera: #如果是第一帧的或者gt_camera=True（这里为false）
                c2w = gt_c2w #第一帧就用真实的位姿吧，后面就用优化的估计位姿了
                if not self.no_vis_on_first_frame:
                    self.visualizer.vis(
                        idx, 0, gt_depth, gt_color, c2w, self.c, self.decoders)

            else: #如果不是第一帧
                gt_camera_tensor = get_tensor_from_camera(gt_c2w) #将当前真实位姿的变换矩阵转换为四元数quad和平移T，赋给gt_camera_tensor
                if self.const_speed_assumption and idx - 2 >= 0: # 如果允许高速估计位姿并且在第三帧以后
                    pre_c2w = pre_c2w.float()
                    delta = pre_c2w @ self.estimate_c2w_list[idx - 2].to(
                        device).float().inverse() #求相对变化量 = 上一帧位姿 / 上两帧位姿
                    estimated_new_cam_c2w = delta @ pre_c2w #当前估计位姿 = 上一帧位姿 * 相对变化量
                else:
                    estimated_new_cam_c2w = pre_c2w #如果在第二帧或第三帧，则假设位姿不变

                camera_tensor = get_tensor_from_camera( #将当前估计位姿的变换矩阵转换为四元数quad和平移T，赋给camera_tensor
                    estimated_new_cam_c2w.detach())
                if self.seperate_LR: #作者不打算分开优化quad和T，设了seperate_LR = false，所以会默认执行else
                    camera_tensor = camera_tensor.to(device).detach()
                    T = camera_tensor[-3:] #取后面的平移T
                    quad = camera_tensor[:4] #取前面的四元数quad
                    cam_para_list_quad = [quad]
                    quad = Variable(quad, requires_grad=True) #封装成Variable，准备自动微分，优化位姿
                    T = Variable(T, requires_grad=True) #新版torch别用Variable，用torch.Tensor
                    camera_tensor = torch.cat([quad, T], 0) #拼接四元数quad和平移T
                    cam_para_list_T = [T] #用list来存T和quad
                    cam_para_list_quad = [quad]
                    optimizer_camera = torch.optim.Adam([{'params': cam_para_list_T, 'lr': self.cam_lr}, #准备优化T和quad
                                                         {'params': cam_para_list_quad, 'lr': self.cam_lr * 0.2}])
                else: #但seperate_LR = false，所以会默认执行下面的else代码，因为作者不打算分开优化quad和T
                    camera_tensor = Variable(
                        camera_tensor.to(device), requires_grad=True)
                    cam_para_list = [camera_tensor] #把quad和T放一起装进list里
                    optimizer_camera = torch.optim.Adam(
                        cam_para_list, lr=self.cam_lr) #把quad和T放一起优化

                initial_loss_camera_tensor = torch.abs( #计算初始的位姿loss = 真实位姿 - 估计位姿 （这里loss也可以改改），后面就优化这个loss
                    gt_camera_tensor.to(device) - camera_tensor).mean().item()
                candidate_cam_tensor = None
                current_min_loss = 10000000000. #1e10
                for cam_iter in range(self.num_cam_iters): #相机迭代跟踪次数，10次
                    if self.seperate_LR: #如果quad和T分开优化
                        camera_tensor = torch.cat([quad, T], 0).to(self.device) #赶紧合并吧

                    self.visualizer.vis(
                        idx, cam_iter, gt_depth, gt_color, camera_tensor, self.c, self.decoders)

                    loss = self.optimize_cam_in_batch(
                        camera_tensor, gt_color, gt_depth, self.tracking_pixels, optimizer_camera)
                        # 迭代一次相机的跟踪操作，对像素点进行采样，渲染深度和颜色，计算两个loss并反向传播，以此进行优化，一次处理batch_size个数据
                        #这里batch_size为tracking_pixels,即200
                    if cam_iter == 0:
                        initial_loss = loss #初次迭代，则记录第一次的渲染loss（渲染loss即为几何loss+颜色loss）

                    loss_camera_tensor = torch.abs(
                        gt_camera_tensor.to(device) - camera_tensor).mean().item() #计算优化后的相机位姿loss
                    if self.verbose:
                        if cam_iter == self.num_cam_iters - 1:
                            print(
                                f'Re-rendering loss: {initial_loss:.2f}->{loss:.2f} ' +
                                f'camera tensor error: {initial_loss_camera_tensor:.4f}->{loss_camera_tensor:.4f}')
                    if loss < current_min_loss:
                        current_min_loss = loss #记录最小渲染loss
                        candidate_cam_tensor = camera_tensor.clone().detach() #将最小渲染loss时的位姿作为候选位姿
                bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
                    [1, 4])).type(torch.float32).to(self.device) #这个bottom没太看懂
                c2w = get_camera_from_tensor(
                    candidate_cam_tensor.clone().detach()) #将quad和T换回变换矩阵RT，得到优化后的位姿
                c2w = torch.cat([c2w, bottom], dim=0)
            self.estimate_c2w_list[idx] = c2w.clone().cpu() #将优化后的位姿放入list里
            self.gt_c2w_list[idx] = gt_c2w.clone().cpu()
            pre_c2w = c2w.clone() #更新上一个位姿
            self.idx[0] = idx
            if self.low_gpu_mem:
                torch.cuda.empty_cache()
