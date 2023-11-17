import torch
from src.common import get_rays, raw2outputs_nerf_color, sample_pdf


class Renderer(object):
    def __init__(self, cfg, args, slam, points_batch_size=500000, ray_batch_size=100000):
        self.ray_batch_size = ray_batch_size #采样射线的batch_size
        self.points_batch_size = points_batch_size #采样点的batch_size

        # 下面几个参数是nerf经常用的，可以从一些nerf的实现代码里面找到
        # 比如我注释的nerf-pytorch: https://github.com/Immortalqx/nerf-pytorch-noted
        self.lindisp = cfg['rendering']['lindisp']  # 根据视差还是深度进行线性采样，True为采用视差,这里为false
        self.perturb = cfg['rendering']['perturb']  # 如果非零，则在分层随机时间点对每条射线进行采样，这里为0
        self.N_samples = cfg['rendering']['N_samples']  # 沿每条射线采样的不同次数，32
        self.N_surface = cfg['rendering']['N_surface']  # 这个和N_importance类似，不过是直接在物体表面附近采样的附加次数，16
        self.N_importance = cfg['rendering']['N_importance']  # 沿每条射线采样的附加次数，0

        self.scale = cfg['scale']  # 未使用，1
        self.occupancy = cfg['occupancy']  # occupancy or volume density
        self.nice = slam.nice  # NICE-SLAM or iMAP
        self.bound = slam.bound #边界

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def eval_points(self, p, decoders, c=None, stage='color', device='cuda:0'): #
        """
        Evaluates the occupancy and/or color value for the points.
        评估神经网络输出的占据值（体密度）和颜色值
        Args:
            p (tensor, N*3): Point coordinates.
            decoders (nn.module decoders): Decoders.
            c (dicts, optional): Feature grids. Defaults to None.
            stage (str, optional): Query stage, corresponds to different levels. Defaults to 'color'.
            device (str, optional): CUDA device. Defaults to 'cuda:0'.

        Returns:
            ret (tensor): occupancy (and color) value of input points.
        """

        p_split = torch.split(p, self.points_batch_size)  # 将输入的点p分批，每批batch_size个,把一个tensor拆分成多个tensor
        bound = self.bound
        rets = []
        for pi in p_split:
            # mask for points out of bound, 过滤掉边界外的点，只考虑边界内的点
            mask_x = (pi[:, 0] < bound[0][1]) & (pi[:, 0] > bound[0][0])
            mask_y = (pi[:, 1] < bound[1][1]) & (pi[:, 1] > bound[1][0])
            mask_z = (pi[:, 2] < bound[2][1]) & (pi[:, 2] > bound[2][0])
            mask = mask_x & mask_y & mask_z  # x,y,z任意一个不在范围内就记False

            pi = pi.unsqueeze(0)  # 扩充了第0维

            if self.nice:
                ret = decoders(pi, c_grid=c, stage=stage)  # 经过decoder得到occupancy或color
            else:
                ret = decoders(pi, c_grid=None)

            ret = ret.squeeze(0)
            if len(ret.shape) == 1 and ret.shape[0] == 4:
                ret = ret.unsqueeze(0)

            ret[~mask, 3] = 100  # 对于超出边界的部分，这里将occupancy赋值成100 | 用100表示需要被忽略掉
            rets.append(ret)

        ret = torch.cat(rets, dim=0)
        return ret

    # 下面这个是类Renderer最重要的函数
    def render_batch_ray(self, c, decoders, rays_d, rays_o, device, stage, gt_depth=None):
        """
        Render color, depth and uncertainty of a batch of rays.
        渲染颜色、深度和一组射线的不确定性
        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            rays_d (tensor, N*3): rays direction.
            rays_o (tensor, N*3): rays origin.
            device (str): device name to compute on.
            stage (str): query stage.
            gt_depth (tensor, optional): sensor depth image. Defaults to None.

        Returns:
            depth (tensor): rendered depth.
            uncertainty (tensor): rendered uncertainty.
            color (tensor): rendered color.
        """

        N_samples = self.N_samples  # 沿每条射线采样的不同次数，32
        N_surface = self.N_surface  # 这个和N_importance类似，不过是直接在物体表面附近采样的附加次数，16
        N_importance = self.N_importance # 0

        N_rays = rays_o.shape[0] #采样射线的数量
        # 每条射线都有其最近采样距离near和最远采样距离far
        # 不能够在边界之外采样，所以这里需要计算一个范围，让z_vals*rays_d待在边界内
        # 这里先计算最近的深度，near
        # TODO 为什么coarse不需要gt_depth，并且将N_surface设置为0？
        #  我觉得是先运行的middle和fine，采样距离需要精细计算，
        #  之后的coarse采样middle和fine没注意到的物体，不需要那么精确，near统一设为0.01
        if stage == 'coarse':
            gt_depth = None #coarse模式不需要真实深度
        if gt_depth is None:
            N_surface = 0  #coarse模式直接在物体表面附近采样的附加次数为0
            near = 0.01   #最近的深度统一设为0.01
        else: #如果是middle或fine
            gt_depth = gt_depth.reshape(-1, 1)  # 将真实深度展开成向量
            gt_depth_samples = gt_depth.repeat(1, N_samples) #代表每条射线采样的深度都为真实深度？
            near = gt_depth_samples * 0.01  # 每条射线的near根据这条射线对应的深度确定

        # 计算最远的深度，far
        with torch.no_grad():
            det_rays_o = rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            det_rays_d = rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            # 这个相当于计算一个间隔，在不超过边界的情况下，从射线原点往后能取多少，往前能取多少
            t = (self.bound.unsqueeze(0).to(device) - det_rays_o) / det_rays_d  # (N, 3, 2)
            # 因为最后是深度值z_vals*det_rays_d的形式，所以xyz任意一个都不能超过边界，只能取最小的范围
            # 由于这里我们的相机只能看见前面的东西，所以就忽略掉了torch.max(t, dim=1)的情况
            far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)  #又要最远又要不能超过边界
            far_bb = far_bb.unsqueeze(-1)
            far_bb += 0.01  # 这里是防止far=0的情况？
        if gt_depth is not None:  #middle或fine
            # in case the bound is too large 防止边界太大
            far = torch.clamp(far_bb, 0, torch.max(gt_depth * 1.2))  # 限制far的范围在[0,torch.max(gt_depth * 1.2)]内
        else:
            far = far_bb

        if N_surface > 0:  # NICE-SLAM为True，对表面采样
            if False:
                # this naive implementation downgrades performance
                # ↑来自作者对这行代码的无情diss↑
                gt_depth_surface = gt_depth.repeat(1, N_surface)
                t_vals_surface = torch.linspace(
                    0., 1., steps=N_surface).to(device)
                z_vals_surface = (0.95 * gt_depth_surface * (1. - t_vals_surface) +
                                  1.05 * gt_depth_surface * (t_vals_surface))
            else:
                # since we want to colorize even on regions with no depth sensor readings,
                # meaning colorize on interpolated geometry region,
                # we sample all pixels (not using depth mask) for color loss.
                # Therefore, for pixels with non-zero depth value, we sample near the surface,
                # since it is not a good idea to sample 16 points near (half even behind) camera,
                # for pixels with zero depth value, we sample uniformly from camera to max_depth.

                # 由于我们希望在没有深度传感器读数的区域上进行着色，
                # 意味着在插值的几何区域上着色，
                # 我们采样所有的像素点来计算颜色损失loss（不使用深度的mask）
                # 因此，对于深度值大于0的像素，我们在深度表面附近采样，
                # 对于深度为零的像素来说，我们从相机（深度为0）到最大深度均匀采样

                # TODO 注意：这段else代码是运用对物体表面采样N_surface次，而非N_sample次

                # 先对有深度的区域进行处理，得到这部分的采样间隔
                gt_none_zero_mask = gt_depth > 0 #有深度区域的mask
                gt_none_zero = gt_depth[gt_none_zero_mask]  #mask掉，得到有深度的区域
                gt_none_zero = gt_none_zero.unsqueeze(-1) #加一维
                gt_depth_surface = gt_none_zero.repeat(1, N_surface) # 额外采样N_surface次
                t_vals_surface = torch.linspace(0., 1., steps=N_surface).double().to(device) #采样间隔
                # 论文里面提到在depth附近+-0.05D的区域进行采样，这里是对应的实现
                # emperical range 0.05*depth
                z_vals_surface_depth_none_zero = (0.95 * gt_depth_surface * (1. - t_vals_surface) +
                                                  1.05 * gt_depth_surface * (t_vals_surface))
                # 保存到最终的z_vals_surface中
                z_vals_surface = torch.zeros(gt_depth.shape[0], N_surface).to(device).double() #初始化
                gt_none_zero_mask = gt_none_zero_mask.squeeze(-1)
                z_vals_surface[gt_none_zero_mask, :] = z_vals_surface_depth_none_zero #存储到最终的z_vals_surface中

                # 再对没有深度的区域进行处理，得到这部分的采样间隔near和far
                # 由于没有深度，就指定near和far的具体数值。
                near_surface = 0.001  #最近采样距离
                far_surface = torch.max(gt_depth)  #最深采样距离far
                z_vals_surface_depth_zero = (near_surface * (1. - t_vals_surface) +
                                             far_surface * (t_vals_surface))  #采样间隔
                # 下面这行代码没有用上吧？这个看起来什么都没有影响？？？
                z_vals_surface_depth_zero.unsqueeze(0).repeat((~gt_none_zero_mask).sum(), 1)
                # 保存到最终的z_vals_surface中，没有深度的那行都会被赋值为z_vals_surface_depth_zero
                z_vals_surface[~gt_none_zero_mask, :] = z_vals_surface_depth_zero


        # TODO 这里是采样N_samples次，而非N_surface次
        t_vals = torch.linspace(0., 1., steps=N_samples, device=device) #linspace(0,1)代表均匀采样


        if not self.lindisp:  # 根据深度near和far均匀采样
            z_vals = near * (1. - t_vals) + far * (t_vals)
        else:  # 根据视差采样（双目）
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

        if self.perturb > 0.:  # 是否在每个间隔内随机采样（而非均匀采样），这里为0
            # 本来z_vals是[a,b,c]
            # 这里给处理成了[x,y,z]，其中x取值范围为[a,(a+b)/2]，y取值范围为[(a+b)/2,(b+c)/2]，z取值范围为[(b+c)/2,c]
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(device)
            z_vals = lower + (upper - lower) * t_rand

        if N_surface > 0: # TODO 将N_sample和N_surface两次采样的结果拼接并排序
            z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_surface.double()], -1), -1)  # 从小到大排序

        # 试着写了一下代码，下面pts的格式是[N_rays, 3, N_samples+N_surface, 1]，可能是我自己写的代码定义有问题？
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples+N_surface, 3]
        pointsf = pts.reshape(-1, 3)
        #下面就放入网络进行训练咯
        raw = self.eval_points(pointsf, decoders, c, stage, device) #放入decoder，得到几何损失和颜色损失以及不确定性
        raw = raw.reshape(N_rays, N_samples + N_surface, -1)

        depth, uncertainty, color, weights = raw2outputs_nerf_color(
            raw, z_vals, rays_d, occupancy=self.occupancy, device=device)

        # NICE-SLAM默认N_importance=0，毕竟论文里说自己不会像NeRF或者iMAP那样查询两次网络，而是一步到位。
        # 如果查询两次的话，上面的权重会被使用到，其他的东西似乎就抛弃了。
        if N_importance > 0:
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(
                z_vals_mid, weights[..., 1:-1], N_importance, det=(self.perturb == 0.), device=device)
            z_samples = z_samples.detach()
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
            pts = pts.reshape(-1, 3)
            raw = self.eval_points(pts, decoders, c, stage, device)
            raw = raw.reshape(N_rays, N_samples + N_importance + N_surface, -1)

            depth, uncertainty, color, weights = raw2outputs_nerf_color(
                raw, z_vals, rays_d, occupancy=self.occupancy, device=device)
            return depth, uncertainty, color

        return depth, uncertainty, color

    def render_img(self, c, decoders, c2w, device, stage, gt_depth=None):
        """
        Renders out depth, uncertainty, and color images.
        渲染深度、不确定性和彩色图像
        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            c2w (tensor): camera to world matrix of current frame.
            device (str): device name to compute on.
            stage (str): query stage.
            gt_depth (tensor, optional): sensor depth image. Defaults to None.

        Returns:
            depth (tensor, H*W): rendered depth image.
            uncertainty (tensor, H*W): rendered uncertainty image.
            color (tensor, H*W*3): rendered color image.
        """
        with torch.no_grad():
            H = self.H
            W = self.W
            rays_o, rays_d = get_rays( #根据位姿和相机内参获取采样射线
                H, W, self.fx, self.fy, self.cx, self.cy, c2w, device)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            depth_list = []
            uncertainty_list = []
            color_list = []

            ray_batch_size = self.ray_batch_size
            gt_depth = gt_depth.reshape(-1)

            for i in range(0, rays_d.shape[0], ray_batch_size):
                rays_d_batch = rays_d[i:i + ray_batch_size]
                rays_o_batch = rays_o[i:i + ray_batch_size] #分批
                if gt_depth is None: #如果没有深度
                    ret = self.render_batch_ray(
                        c, decoders, rays_d_batch, rays_o_batch, device, stage, gt_depth=None)
                else: #有深度
                    gt_depth_batch = gt_depth[i:i + ray_batch_size]
                    ret = self.render_batch_ray(
                        c, decoders, rays_d_batch, rays_o_batch, device, stage, gt_depth=gt_depth_batch)

                depth, uncertainty, color = ret  #得到深度，不确定性和颜色
                depth_list.append(depth.double())
                uncertainty_list.append(uncertainty.double())
                color_list.append(color)

            depth = torch.cat(depth_list, dim=0)
            uncertainty = torch.cat(uncertainty_list, dim=0)
            color = torch.cat(color_list, dim=0)

            depth = depth.reshape(H, W)
            uncertainty = uncertainty.reshape(H, W)
            color = color.reshape(H, W, 3)
            return depth, uncertainty, color

    # this is only for imap*
    def regulation(self, c, decoders, rays_d, rays_o, gt_depth, device, stage='color'):
        """
        Regulation that discourage any geometry from the camera center to 0.85*depth.
        For imap, the geometry will not be as good if this loss is not added.
        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            rays_d (tensor, N*3): rays direction.
            rays_o (tensor, N*3): rays origin.
            gt_depth (tensor): sensor depth image
            device (str): device name to compute on.
            stage (str, optional):  query stage. Defaults to 'color'.

        Returns:
            sigma (tensor, N): volume density of sampled points.
        """
        gt_depth = gt_depth.reshape(-1, 1)
        gt_depth = gt_depth.repeat(1, self.N_samples)
        t_vals = torch.linspace(0., 1., steps=self.N_samples).to(device)
        near = 0.0
        far = gt_depth * 0.85
        z_vals = near * (1. - t_vals) + far * (t_vals)
        perturb = 1.0
        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(device)
            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
              z_vals[..., :, None]  # (N_rays, N_samples, 3)
        pointsf = pts.reshape(-1, 3)
        raw = self.eval_points(pointsf, decoders, c, stage, device)
        sigma = raw[:, -1]
        return sigma
