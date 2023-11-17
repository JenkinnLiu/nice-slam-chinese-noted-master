import numpy as np
import open3d as o3d
import skimage
import torch
import torch.nn.functional as F
import trimesh
from packaging import version
from src.utils.datasets import get_dataset


class Mesher(object):

    def __init__(self, cfg, args, slam, points_batch_size=500000, ray_batch_size=100000):
        """
        Mesher class, given a scene representation, the mesher extracts the mesh from it .
        Mesher类，来用于表征场景，即mesher用于提取特征网格里的mesh
        Args:
            cfg (dict): parsed config dict.
            args (class 'argparse.Namespace'): argparse arguments.
            slam (class NICE-SLAM): NICE-SLAM main class.
            points_batch_size (int): maximum points size for query in one batch. 
                                     Used to alleviate GPU memeory usage. Defaults to 500000.
            ray_batch_size (int): maximum ray size for query in one batch. 
                                  Used to alleviate GPU memeory usage. Defaults to 100000.
        """
        self.points_batch_size = points_batch_size #分批处理point制定batch_size
        self.ray_batch_size = ray_batch_size #分批处理射线
        self.renderer = slam.renderer
        self.coarse = cfg['coarse'] #True,
        self.scale = cfg['scale'] #1
        self.occupancy = cfg['occupancy'] #True，应该是算体密度的

        self.resolution = cfg['meshing']['resolution'] #分辨率256，如果要更精细的话可以512
        self.level_set = cfg['meshing']['level_set'] #0
        self.clean_mesh_bound_scale = cfg['meshing']['clean_mesh_bound_scale'] #1.02，清空网格边界的规模？
        self.remove_small_geometry_threshold = cfg['meshing']['remove_small_geometry_threshold'] #0.2，去除小的表面形状的阈值？
        self.color_mesh_extraction_method = cfg['meshing']['color_mesh_extraction_method'] #提取mesh的方法：direct_point_query
        self.get_largest_components = cfg['meshing']['get_largest_components'] #获得最大的部分,false
        self.depth_test = cfg['meshing']['depth_test'] #深度测试,false

        self.bound = slam.bound #边界
        self.nice = slam.nice
        self.verbose = slam.verbose
        #marching_cubes_bound: [[0.0,6.5],[0.0,4.0],[0,3.5]]
        self.marching_cubes_bound = torch.from_numpy( #重建的三维cube的边界
            np.array(cfg['mapping']['marching_cubes_bound']) * self.scale)

        self.frame_reader = get_dataset(cfg, args, self.scale, device='cpu') #DataLoader
        self.n_img = len(self.frame_reader) #图片的数量

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def point_masks(self, input_points, keyframe_dict, estimate_c2w_list,
                    idx, device, get_mask_use_all_frames=False):
        """
        Split the input points into seen, unseen, and forcast,
        according to the estimated camera pose and depth image.
        #根据估计位姿和深度图，将输入点分为可见点、不可见点和待预测点这三类
        Args:
            input_points (tensor): input points.
            keyframe_dict (list): list of keyframe info dictionary.
            estimate_c2w_list (tensor): estimated camera pose.
            idx (int): current frame index.
            device (str): device name to compute on.

        Returns:
            seen_mask (tensor): the mask for seen area.
            forecast_mask (tensor): the mask for forecast area.
            unseen_mask (tensor): the mask for unseen area.
        """
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        if not isinstance(input_points, torch.Tensor): #通过torch.from_numpy转换为torch.Tensor张量
            input_points = torch.from_numpy(input_points)
        input_points = input_points.clone().detach()
        seen_mask_list = [] #可见点的mask
        forecast_mask_list = [] #预测点的mask
        unseen_mask_list = [] #不可见点的masl
        for i, pnts in enumerate( #遍历输入点input_points
                torch.split(input_points, self.points_batch_size, dim=0)):
            points = pnts.to(device).float()
            # should divide the points into three parts, seen and forecast and unseen
            # seen: union of all the points in the viewing frustum of keyframes
            # forecast: union of all the points in the extended edge of the viewing frustum of keyframes
            # unseen: all the other points
            # 应该把要点分为三个部分，看得见的、待预测的和看不见的
            # seen可见点:关键帧中所看到的平截面中所有点的并集
            # forecast待预测点：关键帧中所看到的平截面的扩展边缘中的所有点的并集
            # unseen不可见点：所有其他点

            #虽然下面的暂时看不懂，但是也挺重要的，因为它规定了将点分为待预测的点的条件，待预测的点填补空缺（回忆回忆论文里的那张蓝色的图，可以填补大空缺的）
            seen_mask = torch.zeros((points.shape[0])).bool().to(device) #初始化mask
            forecast_mask = torch.zeros((points.shape[0])).bool().to(device)
            if get_mask_use_all_frames: #false，指利用所有帧来求mask
                for i in range(0, idx + 1, 1):
                    c2w = estimate_c2w_list[i].cpu().numpy() #取出当前下标为i的位姿变换矩阵c2w
                    w2c = np.linalg.inv(c2w) #c2w的逆
                    w2c = torch.from_numpy(w2c).to(device).float()
                    ones = torch.ones_like(
                        points[:, 0]).reshape(-1, 1).to(device)
                    homo_points = torch.cat([points, ones], dim=1).reshape(
                        -1, 4, 1).to(device).float()  # (N, 4)
                    # (N, 4, 1)=(4,4)*(N, 4, 1)
                    cam_cord_homo = w2c @ homo_points
                    cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)

                    K = torch.from_numpy(
                        np.array([[fx, .0, cx], [.0, fy, cy],
                                  [.0, .0, 1.0]]).reshape(3, 3)).to(device)
                    cam_cord[:, 0] *= -1
                    uv = K.float() @ cam_cord.float()
                    z = uv[:, -1:] + 1e-8
                    uv = uv[:, :2] / z
                    uv = uv.float()
                    edge = 0
                    cur_mask_seen = (uv[:, 0] < W - edge) & (
                            uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
                    cur_mask_seen = cur_mask_seen & (z[:, :, 0] < 0)

                    edge = -1000
                    cur_mask_forecast = (uv[:, 0] < W - edge) & (
                            uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
                    cur_mask_forecast = cur_mask_forecast & (z[:, :, 0] < 0)

                    # forecast
                    cur_mask_forecast = cur_mask_forecast.reshape(-1)
                    # seen
                    cur_mask_seen = cur_mask_seen.reshape(-1)

                    seen_mask |= cur_mask_seen
                    forecast_mask |= cur_mask_forecast
            else: #默认进入else，通过单个帧求三个mask
                for keyframe in keyframe_dict: #遍历每一帧
                    c2w = keyframe['est_c2w'].cpu().numpy() #当前帧的位姿
                    w2c = np.linalg.inv(c2w) #求逆
                    w2c = torch.from_numpy(w2c).to(device).float()
                    # 用于计算相机坐标系下的点在图像平面上的投影位置（3d转2d），并根据投影位置生成mask。
                    # 下面的代码实在看不懂捏
                    ones = torch.ones_like(
                        points[:, 0]).reshape(-1, 1).to(device)
                    homo_points = torch.cat([points, ones], dim=1).reshape(
                        -1, 4, 1).to(device).float()
                    cam_cord_homo = w2c @ homo_points
                    cam_cord = cam_cord_homo[:, :3]

                    K = torch.from_numpy(
                        np.array([[fx, .0, cx], [.0, fy, cy],
                                  [.0, .0, 1.0]]).reshape(3, 3)).to(device)
                    cam_cord[:, 0] *= -1
                    uv = K.float() @ cam_cord.float()
                    z = uv[:, -1:] + 1e-8
                    uv = uv[:, :2] / z
                    uv = uv.float()
                    edge = 0
                    cur_mask_seen = (uv[:, 0] < W - edge) & (
                            uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
                    cur_mask_seen = cur_mask_seen & (z[:, :, 0] < 0)

                    edge = -1000
                    cur_mask_forecast = (uv[:, 0] < W - edge) & (
                            uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
                    cur_mask_forecast = cur_mask_forecast & (z[:, :, 0] < 0)

                    if self.depth_test: #如果有深度测试的话
                        gt_depth = keyframe['depth'].to(
                            device).reshape(1, 1, H, W) #加载真实深度
                        vgrid = uv.reshape(1, 1, -1, 2)
                        # normalized to [-1, 1],即归一化
                        vgrid[..., 0] = (vgrid[..., 0] / (W - 1) * 2.0 - 1.0)
                        vgrid[..., 1] = (vgrid[..., 1] / (H - 1) * 2.0 - 1.0)
                        depth_sample = F.grid_sample( #在深度图中采样深度特征
                            gt_depth, vgrid, padding_mode='zeros', align_corners=True)
                        depth_sample = depth_sample.reshape(-1)
                        max_depth = torch.max(depth_sample) #求得采样深度的最大深度
                        # forecast
                        cur_mask_forecast = cur_mask_forecast.reshape(-1)
                        proj_depth_forecast = -cam_cord[cur_mask_forecast,
                        2].reshape(-1)
                        cur_mask_forecast[cur_mask_forecast.clone()] &= proj_depth_forecast < max_depth
                        # seen，求得cur_mask_seen, proj_depth_seen
                        cur_mask_seen = cur_mask_seen.reshape(-1)
                        proj_depth_seen = - cam_cord[cur_mask_seen, 2].reshape(-1)
                        cur_mask_seen[cur_mask_seen.clone()] &= \
                            (proj_depth_seen < depth_sample[cur_mask_seen] + 2.4) \
                            & (depth_sample[cur_mask_seen] - 2.4 < proj_depth_seen)
                    else:
                        max_depth = torch.max(keyframe['depth']) * 1.1 #最大深度

                        # forecast
                        cur_mask_forecast = cur_mask_forecast.reshape(-1)
                        proj_depth_forecast = -cam_cord[cur_mask_forecast,
                        2].reshape(-1)
                        cur_mask_forecast[
                            cur_mask_forecast.clone()] &= proj_depth_forecast < max_depth

                        # seen
                        cur_mask_seen = cur_mask_seen.reshape(-1)
                        proj_depth_seen = - \
                            cam_cord[cur_mask_seen, 2].reshape(-1)
                        cur_mask_seen[cur_mask_seen.clone(
                        )] &= proj_depth_seen < max_depth

                    seen_mask |= cur_mask_seen
                    forecast_mask |= cur_mask_forecast

            forecast_mask &= ~seen_mask
            unseen_mask = ~(seen_mask | forecast_mask)

            seen_mask = seen_mask.cpu().numpy()
            forecast_mask = forecast_mask.cpu().numpy()
            unseen_mask = unseen_mask.cpu().numpy()

            seen_mask_list.append(seen_mask)
            forecast_mask_list.append(forecast_mask)
            unseen_mask_list.append(unseen_mask)

        seen_mask = np.concatenate(seen_mask_list, axis=0)
        forecast_mask = np.concatenate(forecast_mask_list, axis=0)
        unseen_mask = np.concatenate(unseen_mask_list, axis=0)
        return seen_mask, forecast_mask, unseen_mask

    def get_bound_from_frames(self, keyframe_dict, scale=1):
        """
        Get the scene bound (convex hull),
        using sparse estimated camera poses and corresponding depth images.
        计算场景的边界（凸包）,使用稀疏估计的相机姿态和相应的深度图像。
        Args:
            keyframe_dict (list): list of keyframe info dictionary.
            scale (float): scene scale.

        Returns:
            return_mesh (trimesh.Trimesh): the convex hull.
        """

        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        if version.parse(o3d.__version__) >= version.parse('0.13.0'): #作者挺贴心的，还照顾低版本
            # for new version as provided in environment.yaml
            volume = o3d.pipelines.integration.ScalableTSDFVolume( #这里就是open3d的东西了，初始化volume
                voxel_length=4.0 * scale / 512.0,
                sdf_trunc=0.04 * scale,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        else:
            # for lower version
            volume = o3d.integration.ScalableTSDFVolume(
                voxel_length=4.0 * scale / 512.0,
                sdf_trunc=0.04 * scale,
                color_type=o3d.integration.TSDFVolumeColorType.RGB8)
        cam_points = []
        for keyframe in keyframe_dict: #遍历每一帧
            c2w = keyframe['est_c2w'].cpu().numpy()
            # convert to open3d camera pose #转换为open3d的相机位姿
            c2w[:3, 1] *= -1.0  #前四行第二列取负数
            c2w[:3, 2] *= -1.0  #前四行第三列取负数
            w2c = np.linalg.inv(c2w)
            cam_points.append(c2w[:3, 3])  #相机位姿
            depth = keyframe['depth'].cpu().numpy()
            color = keyframe['color'].cpu().numpy()

            depth = o3d.geometry.Image(depth.astype(np.float32)) #得到open3d的深度
            color = o3d.geometry.Image(np.array( #open3d的颜色
                (color * 255).astype(np.uint8)))

            intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy) #open3d的相机内参
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(  #open3d的RGBD图片
                color,
                depth,
                depth_scale=1,
                depth_trunc=1000,
                convert_rgb_to_intensity=False)
            volume.integrate(rgbd, intrinsic, w2c)  #2D转3D

        cam_points = np.stack(cam_points, axis=0)
        mesh = volume.extract_triangle_mesh()
        mesh_points = np.array(mesh.vertices)
        points = np.concatenate([cam_points, mesh_points], axis=0)
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)) #根据RGBD建立3D点云
        mesh, _ = o3d_pc.compute_convex_hull()  #计算边界（凸包）
        mesh.compute_vertex_normals()
        if version.parse(o3d.__version__) >= version.parse('0.13.0'):
            mesh = mesh.scale(self.clean_mesh_bound_scale, mesh.get_center())
        else:
            mesh = mesh.scale(self.clean_mesh_bound_scale, center=True)
        points = np.array(mesh.vertices)  #点
        faces = np.array(mesh.triangles)  #平截面
        return_mesh = trimesh.Trimesh(vertices=points, faces=faces)
        return return_mesh  #返回的mesh

    def eval_points(self, p, decoders, c=None, stage='color', device='cuda:0'):
        """
        Evaluates the occupancy and/or color value for the points.
        评估神经网络输出的占据值（体密度）和颜色值
        Args:
            p (tensor, N*3): point coordinates.
            decoders (nn.module decoders): decoders.
            c (dicts, optional): feature grids. Defaults to None.
            stage (str, optional): query stage, corresponds to different levels. Defaults to 'color'.
            device (str, optional): device name to compute on. Defaults to 'cuda:0'.

        Returns:
            ret (tensor): occupancy (and color) value of input points.
        """

        p_split = torch.split(p, self.points_batch_size)  #将输入的点p分批，每批batch_size个,把一个tensor拆分成多个tensor
        bound = self.bound #边界
        rets = []
        for pi in p_split:  #遍历每批点p
            # mask for points out of bound， 过滤掉边界外的点，只考虑边界内的点
            mask_x = (pi[:, 0] < bound[0][1]) & (pi[:, 0] > bound[0][0])
            mask_y = (pi[:, 1] < bound[1][1]) & (pi[:, 1] > bound[1][0])
            mask_z = (pi[:, 2] < bound[2][1]) & (pi[:, 2] > bound[2][0])
            mask = mask_x & mask_y & mask_z

            pi = pi.unsqueeze(0)  #扩充了第0维
            if self.nice:
                ret = decoders(pi, c_grid=c, stage=stage)  #放入神经网络,得到occupancy或color
            else:
                ret = decoders(pi, c_grid=None)
            ret = ret.squeeze(0)
            if len(ret.shape) == 1 and ret.shape[0] == 4:
                ret = ret.unsqueeze(0)

            ret[~mask, 3] = 100 # 对于超出边界的部分，这里将occupancy赋值成100 | 用100表示需要被忽略掉
            rets.append(ret)

        ret = torch.cat(rets, dim=0)
        return ret

    def get_grid_uniform(self, resolution):
        """
        Get query point coordinates for marching cubes.
        输入边界和分辨率，构造很多细小的三维网格点
        Args:
            resolution (int): marching cubes resolution.
            输入分辨率
        Returns:
            (dict): points coordinates and sampled coordinates for each axis.
        """
        bound = self.marching_cubes_bound # 边界

        padding = 0.05
        x = np.linspace(bound[0][0] - padding, bound[0][1] + padding,
                        resolution)
        y = np.linspace(bound[1][0] - padding, bound[1][1] + padding,
                        resolution)
        z = np.linspace(bound[2][0] - padding, bound[2][1] + padding,
                        resolution)

        xx, yy, zz = np.meshgrid(x, y, z) #构造网格
        grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
        grid_points = torch.tensor(np.vstack(
            [xx.ravel(), yy.ravel(), zz.ravel()]).T,
                                   dtype=torch.float)

        return {"grid_points": grid_points, "xyz": [x, y, z]}

    def get_mesh(self,
                 mesh_out_file,
                 c,
                 decoders,
                 keyframe_dict,
                 estimate_c2w_list,
                 idx,
                 device='cuda:0',
                 show_forecast=False,
                 color=True,
                 clean_mesh=True,
                 get_mask_use_all_frames=False):
        """
        Extract mesh from scene representation and save mesh to file.
        从表征的场景中提取mesh并保存至文件
        Args:
            mesh_out_file (str): output mesh filename.
            c (dicts): feature grids.
            decoders (nn.module): decoders.
            keyframe_dict (list):  list of keyframe info.
            estimate_c2w_list (tensor): estimated camera pose.
            idx (int): current processed camera ID.
            device (str, optional): device name to compute on. Defaults to 'cuda:0'.
            show_forecast (bool, optional): show forecast. Defaults to False.
            color (bool, optional): whether to extract colored mesh. Defaults to True.
            clean_mesh (bool, optional): whether to clean the output mesh 
                                        (remove outliers outside the convexhull and small geometry noise). 
                                        Defaults to True.
            get_mask_use_all_frames (bool, optional): 
                whether to use all frames or just keyframes when getting the seen/unseen mask. Defaults to False.
        """
        with torch.no_grad():

            grid = self.get_grid_uniform(self.resolution) #初始化很多细小的有边界的三维网格点
            points = grid['grid_points']  #用points来表示每个网格点
            points = points.to(device)

            if show_forecast: #显示预测点，默认为false，执行下面else的代码
                # 根据估计位姿和深度图，将输入点分为可见点、不可见点和待预测点这三类，得到三种mask
                seen_mask, forecast_mask, unseen_mask = self.point_masks(
                    points, keyframe_dict, estimate_c2w_list, idx, device=device,
                    get_mask_use_all_frames=get_mask_use_all_frames)

                forecast_points = points[forecast_mask] #通过mask得到待预测的点points
                seen_points = points[seen_mask] #通过mask得到可见的points

                z_forecast = []
                for i, pnts in enumerate(
                        torch.split(forecast_points, #遍历所有待预测的点
                                    self.points_batch_size,
                                    dim=0)):
                    z_forecast.append(
                        self.eval_points(pnts, decoders, c, 'coarse',
                                         device).cpu().numpy()[:, -1]) # 放进decoder的coarse层来预测，以此填补空缺
                z_forecast = np.concatenate(z_forecast, axis=0)
                z_forecast += 0.2  #加上0.2不清楚是为啥

                z_seen = []
                for i, pnts in enumerate(  #遍历可见点
                        torch.split(seen_points, self.points_batch_size,
                                    dim=0)):
                    z_seen.append(
                        self.eval_points(pnts, decoders, c, 'fine', #放进fine层，来细化一下
                                         device).cpu().numpy()[:, -1])
                z_seen = np.concatenate(z_seen, axis=0)

                z = np.zeros(points.shape[0])
                z[seen_mask] = z_seen
                z[forecast_mask] = z_forecast
                z[unseen_mask] = -100

            else: #因为是false，所以默认执行else的代码，不预测了，感觉作者做了实验，发现分成三类mask预测效果不好
                mesh_bound = self.get_bound_from_frames( # 计算场景的边界（凸包）,使用稀疏估计的相机姿态和相应的深度图像,返回mesh。
                    keyframe_dict, self.scale)
                z = []
                mask = []
                for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                    mask.append(mesh_bound.contains(pnts.cpu().numpy())) #mask的作用是过滤边界之外的点
                mask = np.concatenate(mask, axis=0)
                for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                    z.append(self.eval_points(pnts, decoders, c, 'fine', #所有mesh顶点都放进fine层，来细化一下,获取几何信息
                                              device).cpu().numpy()[:, -1])

                z = np.concatenate(z, axis=0) #输出fine层的占据值（体密度）放到z里
                z[~mask] = 100 #过滤边界之外的点

            z = z.astype(np.float32)

            try:
                if version.parse(
                        skimage.__version__) > version.parse('0.15.0'):
                    # for new version as provided in environment.yaml
                    verts, faces, normals, values = skimage.measure.marching_cubes( #从3d网格中提取三维等值面
                        volume=z.reshape(
                            grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                            grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                        level=self.level_set,
                        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                                 grid['xyz'][1][2] - grid['xyz'][1][1],
                                 grid['xyz'][2][2] - grid['xyz'][2][1]))
                else:
                    # for lower version
                    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
                        volume=z.reshape(
                            grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                            grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                        level=self.level_set,
                        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                                 grid['xyz'][1][2] - grid['xyz'][1][1],
                                 grid['xyz'][2][2] - grid['xyz'][2][1]))
            except:
                print(
                    'marching_cubes error. Possibly no surface extracted from the level set.'
                )
                return

            # convert back to world coordinates #把提取出的三维等值面转换回去到世界坐标grid
            vertices = verts + np.array(
                [grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

            if clean_mesh: # 清空小mesh，默认为true
                if show_forecast: #false，运行下面的else吧
                    points = vertices
                    mesh = trimesh.Trimesh(vertices=vertices, #用Trimesh包计算三角网格
                                           faces=faces,
                                           process=False)
                    mesh_bound = self.get_bound_from_frames(
                        keyframe_dict, self.scale)
                    contain_mask = []
                    for i, pnts in enumerate(
                            np.array_split(points, self.points_batch_size,
                                           axis=0)):
                        contain_mask.append(mesh_bound.contains(pnts))
                    contain_mask = np.concatenate(contain_mask, axis=0)
                    not_contain_mask = ~contain_mask
                    face_mask = not_contain_mask[mesh.faces].all(axis=1)
                    mesh.update_faces(~face_mask)
                else:
                    points = vertices
                    mesh = trimesh.Trimesh(vertices=vertices, #用Trimesh包计算三角网格
                                           faces=faces,
                                           process=False)
                    seen_mask, forecast_mask, unseen_mask = self.point_masks( #这里求三类mask
                        points, keyframe_dict, estimate_c2w_list, idx, device=device,
                        get_mask_use_all_frames=get_mask_use_all_frames)
                    unseen_mask = ~seen_mask
                    face_mask = unseen_mask[mesh.faces].all(axis=1)
                    mesh.update_faces(~face_mask) #更新可见点和待预测点的三角网格mesh，不更新不可见点的mesh

                # get connected components #获取连接的部分的mesh
                components = mesh.split(only_watertight=False)
                if self.get_largest_components: #false ,运行下面的else吧
                    areas = np.array([c.area for c in components], dtype=np.float)
                    mesh = components[areas.argmax()]
                else:
                    new_components = []
                    for comp in components: #遍历每一个mesh
                        if comp.area > self.remove_small_geometry_threshold * self.scale * self.scale: #comp.area>0.2
                            new_components.append(comp)
                    mesh = trimesh.util.concatenate(new_components) #得到新的，面积更好的mesh，即面积大于0.2
                vertices = mesh.vertices
                faces = mesh.faces

            if color: #true，肯定要搞搞颜色
                if self.color_mesh_extraction_method == 'direct_point_query': #默认，直接获取点
                    # color is extracted by passing the coordinates of mesh vertices through the network
                    # 通过神经网络decoder传递mesh网格的顶点的坐标来提取颜色
                    points = torch.from_numpy(vertices)
                    z = []
                    for i, pnts in enumerate(
                            torch.split(points, self.points_batch_size, dim=0)): #将顶点放入颜色网络来提取颜色
                        z_color = self.eval_points(
                            pnts.to(device).float(), decoders, c, 'color',
                            device).cpu()[..., :3]
                        z.append(z_color)
                    z = torch.cat(z, axis=0)
                    vertex_colors = z.numpy()

                elif self.color_mesh_extraction_method == 'render_ray_along_normal':
                    # for imap*
                    # render out the color of the ray along vertex normal, and assign it to vertex color
                    import open3d as o3d
                    mesh = o3d.geometry.TriangleMesh(
                        vertices=o3d.utility.Vector3dVector(vertices),
                        triangles=o3d.utility.Vector3iVector(faces))
                    mesh.compute_vertex_normals()
                    vertex_normals = np.asarray(mesh.vertex_normals)
                    rays_d = torch.from_numpy(vertex_normals).to(device)
                    sign = -1.0
                    length = 0.1
                    rays_o = torch.from_numpy(
                        vertices + sign * length * vertex_normals).to(device)
                    color_list = []
                    batch_size = self.ray_batch_size
                    gt_depth = torch.zeros(vertices.shape[0]).to(device)
                    gt_depth[:] = length
                    for i in range(0, rays_d.shape[0], batch_size):
                        rays_d_batch = rays_d[i:i + batch_size]
                        rays_o_batch = rays_o[i:i + batch_size]
                        gt_depth_batch = gt_depth[i:i + batch_size]
                        depth, uncertainty, color = self.renderer.render_batch_ray(
                            c, decoders, rays_d_batch, rays_o_batch, device,
                            stage='color', gt_depth=gt_depth_batch)
                        color_list.append(color)
                    color = torch.cat(color_list, dim=0)
                    vertex_colors = color.cpu().numpy()

                vertex_colors = np.clip(vertex_colors, 0, 1) * 255
                vertex_colors = vertex_colors.astype(np.uint8)

                # cyan color for forecast region
                # 论文中预测区域呈现青色
                if show_forecast:
                    seen_mask, forecast_mask, unseen_mask = self.point_masks(
                        vertices, keyframe_dict, estimate_c2w_list, idx, device=device,
                        get_mask_use_all_frames=get_mask_use_all_frames)
                    vertex_colors[forecast_mask, 0] = 0
                    vertex_colors[forecast_mask, 1] = 255
                    vertex_colors[forecast_mask, 2] = 255

            else:
                vertex_colors = None

            vertices /= self.scale
            mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors) #三角网格
            mesh.export(mesh_out_file)
            if self.verbose:
                print('Saved mesh at', mesh_out_file) #存到文件里
