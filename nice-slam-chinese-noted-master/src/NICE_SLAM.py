import os
import time

import numpy as np
import torch
import torch.multiprocessing
import torch.multiprocessing as mp

from src import config
from src.Mapper import Mapper
from src.Tracker import Tracker
from src.utils.datasets import get_dataset
from src.utils.Logger import Logger
from src.utils.Mesher import Mesher
from src.utils.Renderer import Renderer

torch.multiprocessing.set_sharing_strategy('file_system')


class NICE_SLAM():
    """
    NICE_SLAM main class.
    Mainly allocate shared resources, and dispatch mapping and tracking process.
    """

    def __init__(self, cfg, args):

        self.cfg = cfg
        self.args = args  # 未使用
        self.nice = args.nice

        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']  # 未使用
        self.low_gpu_mem = cfg['low_gpu_mem']
        self.verbose = cfg['verbose']
        self.dataset = cfg['dataset']  # 未使用
        self.coarse_bound_enlarge = cfg['model']['coarse_bound_enlarge']

        if args.output is None:
            self.output = cfg['data']['output']
        else:
            self.output = args.output
        self.ckptsdir = os.path.join(self.output, 'ckpts')
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(self.ckptsdir, exist_ok=True)
        os.makedirs(f'{self.output}/mesh', exist_ok=True)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        # 预处理会进行缩放或者裁剪,而这会影响相机内参，所以需要更新一下内参
        self.update_cam()

        # 根据配置文件构建网络模型
        model = config.get_model(cfg, nice=self.nice)
        self.shared_decoders = model

        self.scale = cfg['scale']

        self.load_bound(cfg)  # 加载XYZ轴边界
        if self.nice:
            self.load_pretrain(cfg)  # 加载预训练参数到shared_decoders中
            self.grid_init(cfg)  # 初始化了hierarchical feature网格
        else:
            self.shared_c = {}

        # 父进程启动一个新的Python解释器进程。子进程只会继承那些运行进程对象的 run() 方法所需的资源。
        # 参考链接：https://blog.csdn.net/YNNAD1997/article/details/113829532
        # need to use spawn
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        # 下面使用了大量的share_memory，它允许数据处于一种特殊的状态，可以在不需要拷贝的情况下，任何进程都可以直接使用该数据。
        self.frame_reader = get_dataset(cfg, args, self.scale)  # Data loader
        self.n_img = len(self.frame_reader)  # 图片的数量
        self.estimate_c2w_list = torch.zeros((self.n_img, 4, 4))  # NICE-SLAM估计的相机位姿，Mapper、Tracker、Logger共享
        self.estimate_c2w_list.share_memory_()
        self.gt_c2w_list = torch.zeros((self.n_img, 4, 4))  # 相机位姿真值，Tracker、Logger共享（Tracker赋值，Logger打印）
        self.gt_c2w_list.share_memory_()
        self.idx = torch.zeros((1)).int()  # Tracking线程当前帧id，方便Mapper和Tracker同步（Tracker修改，Mapper读取）
        self.idx.share_memory_()
        self.mapping_first_frame = torch.zeros((1)).int()  # 共享给Mapper，标志第一帧是否处理完毕
        self.mapping_first_frame.share_memory_()
        # the id of the newest frame Mapper is processing
        self.mapping_idx = torch.zeros((1)).int()  # Mapping线程当前帧id，方便Mapper和Tracker同步（Tracker修改，Mapper读取）
        self.mapping_idx.share_memory_()
        self.mapping_cnt = torch.zeros((1)).int()  # counter for mapping
        self.mapping_cnt.share_memory_()

        for key, val in self.shared_c.items():  # 只有iMAP模式才进行这个循环
            val = val.to(self.cfg['mapping']['device'])
            val.share_memory_()
            self.shared_c[key] = val

        self.shared_decoders = self.shared_decoders.to(
            self.cfg['mapping']['device'])
        self.shared_decoders.share_memory()

        # 初始化NICE-SLAM的几个关键组件
        self.renderer = Renderer(cfg, args, self)
        self.mesher = Mesher(cfg, args, self)
        self.logger = Logger(cfg, args, self)
        self.mapper = Mapper(cfg, args, self, coarse_mapper=False)
        if self.coarse:
            self.coarse_mapper = Mapper(cfg, args, self, coarse_mapper=True)
        self.tracker = Tracker(cfg, args, self)

        self.print_output_desc()

    def print_output_desc(self):
        print(f"INFO: The output folder is {self.output}")
        if 'Demo' in self.output:
            print(
                f"INFO: The GT, generated and residual depth/color images can be found under " +
                f"{self.output}/vis/")
        else:
            print(
                f"INFO: The GT, generated and residual depth/color images can be found under " +
                f"{self.output}/tracking_vis/ and {self.output}/mapping_vis/")
        print(f"INFO: The mesh can be found under {self.output}/mesh/")
        print(f"INFO: The checkpoint can be found under {self.output}/ckpt/")

    def update_cam(self):
        """
        Update the camera intrinsics according to pre-processing config, 
        such as resize or edge crop.
        """
        # resize the input images to crop_size (variable name used in lietorch)
        if 'crop_size' in self.cfg['cam']:  # 是否进行缩放，即调整分辨率
            crop_size = self.cfg['cam']['crop_size']
            sx = crop_size[1] / self.W
            sy = crop_size[0] / self.H
            self.fx = sx * self.fx
            self.fy = sy * self.fy
            self.cx = sx * self.cx
            self.cy = sy * self.cy
            self.W = crop_size[1]
            self.H = crop_size[0]

        # croping will change H, W, cx, cy, so need to change here
        if self.cfg['cam']['crop_edge'] > 0:  # 是否进行裁切，注意裁切不影响焦距
            self.H -= self.cfg['cam']['crop_edge'] * 2
            self.W -= self.cfg['cam']['crop_edge'] * 2
            self.cx -= self.cfg['cam']['crop_edge']
            self.cy -= self.cfg['cam']['crop_edge']

    def load_bound(self, cfg):
        """
        Pass the scene bound parameters to different decoders and self.

        Args:
            cfg (dict): parsed config dict.
        """
        # scale the bound if there is a global scaling factor
        self.bound = torch.from_numpy(np.array(cfg['mapping']['bound']) * self.scale)
        #bound是3*4的二维数组，3是xyz坐标，4是4个decoder
        #mapping.bound在nice-slam.yaml里找不到，可能是在命令行里输入的数据集的yaml里有
        # enlarge the bound a bit to allow it divisible by bound_divisible
        bound_divisible = cfg['grid_len']['bound_divisible'] #显示边界
        self.bound[:, 1] = (((self.bound[:, 1] - self.bound[:, 0]) /
                             bound_divisible).int() + 1) * bound_divisible + self.bound[:, 0]

        if self.nice:
            self.shared_decoders.bound = self.bound
            self.shared_decoders.middle_decoder.bound = self.bound
            self.shared_decoders.fine_decoder.bound = self.bound
            self.shared_decoders.color_decoder.bound = self.bound
            if self.coarse:
                self.shared_decoders.coarse_decoder.bound = self.bound * self.coarse_bound_enlarge

    def load_pretrain(self, cfg):
        """
        Load parameters of pretrained ConvOnet checkpoints to the decoders.

        Args:
            cfg (dict): parsed config dict
        """

        # TODO 对模型加载还不太了解，暂时还没去查文档
        if self.coarse: #如果包含coarse层
            ckpt = torch.load(cfg['pretrained_decoders']['coarse'], #加载coarse层参数
                              map_location=cfg['mapping']['device'])
            coarse_dict = {} #加载参数
            for key, val in ckpt['model'].items():
                if ('decoder' in key) and ('encoder' not in key):
                    key = key[8:]
                    coarse_dict[key] = val
            self.shared_decoders.coarse_decoder.load_state_dict(coarse_dict)

        ckpt = torch.load(cfg['pretrained_decoders']['middle_fine'],
                          map_location=cfg['mapping']['device'])
        middle_dict = {}
        fine_dict = {}
        for key, val in ckpt['model'].items():
            if ('decoder' in key) and ('encoder' not in key):
                if 'coarse' in key:
                    key = key[8 + 7:]
                    middle_dict[key] = val
                elif 'fine' in key:
                    key = key[8 + 5:]
                    fine_dict[key] = val
        self.shared_decoders.middle_decoder.load_state_dict(middle_dict)
        self.shared_decoders.fine_decoder.load_state_dict(fine_dict)

    def grid_init(self, cfg):
        """
        Initialize the hierarchical feature grids.
        初始化分层特征网格

        Args:
            cfg (dict): parsed config dict.
        """

        # 加载coarse、middle、fine三个特征网格的网格长度
        # 在论文中，coarse是2m、middle是32cm、fine是16cm，但是在TUM RGB-D数据集上middle是16cm、fine是8cm
        if self.coarse:
            coarse_grid_len = cfg['grid_len']['coarse']
            self.coarse_grid_len = coarse_grid_len
        middle_grid_len = cfg['grid_len']['middle']
        self.middle_grid_len = middle_grid_len
        fine_grid_len = cfg['grid_len']['fine']
        self.fine_grid_len = fine_grid_len
        color_grid_len = cfg['grid_len']['color']
        self.color_grid_len = color_grid_len

        c = {}  # 保存coarse、middle、fine三个特征网格
        c_dim = cfg['model']['c_dim']  # 特征网格上每个特征的维度，这里为32
        # bound是3*4的矩阵，xyz_len=bound的第二列 - bound的第一列，xyz_len为3*1
        xyz_len = self.bound[:, 1] - self.bound[:, 0]  # xyz轴各自的长度，方便后面计算需要多少个网格

        # If you have questions regarding the swap of axis 0 and 2,
        # please refer to https://github.com/cvg/nice-slam/issues/24

        # 为什么需要交换？
        # 因为val_shape的顺序是XYZ，而特征网格的形状应该是[B, C, D, H, W]，DHW分辨对应ZYX
        # B这里不知道含义，但数值应该等于1；C就是c_dim，代表每个特征的维度

        # 下面随机初始化了特征网格
        if self.coarse:
            coarse_key = 'grid_coarse'
            # 这里为什么要enlarge coarse bound，是为了更好的对未观测区域进行预测吗？
            # 或者是因为coarse bound的grid_len太大了，这里又要取整数，不扩展就可能因为取整
            # 下面得到的是[x,y,z]格式的数据，并且数据类型是int
            # 比如tensor([6.7200, 4.1600, 3.5200], dtype=torch.float64)*2/2 得到 [6,4,3]

            # coarse_bound_enlarge=2,coarse_grid_len=2(m)
            # coarse网格的维度（数量）=坐标轴长度*coarse边界的长度/每一个coarse网格的长度
            coarse_val_shape = list(
                map(int, (xyz_len * self.coarse_bound_enlarge / coarse_grid_len).tolist()))
            coarse_val_shape[0], coarse_val_shape[2] = coarse_val_shape[2], coarse_val_shape[0]
            #coarse_val_shape是list类型，存着map（1,3*1），代表每个粗糙特征网格的形状大小
            self.coarse_val_shape = coarse_val_shape
            # 初始化coarse特征网格的值
            val_shape = [1, c_dim, *coarse_val_shape] #【1，32，coarse_val_shape】
            coarse_val = torch.zeros(val_shape).normal_(mean=0, std=0.01)
            c[coarse_key] = coarse_val#在c数组通过key来查询对应粗糙网格的值

        middle_key = 'grid_middle'
        middle_val_shape = list(map(int, (xyz_len / middle_grid_len).tolist()))
        middle_val_shape[0], middle_val_shape[2] = middle_val_shape[2], middle_val_shape[0]
        self.middle_val_shape = middle_val_shape
        val_shape = [1, c_dim, *middle_val_shape]
        middle_val = torch.zeros(val_shape).normal_(mean=0, std=0.01)
        c[middle_key] = middle_val

        fine_key = 'grid_fine'
        fine_val_shape = list(map(int, (xyz_len / fine_grid_len).tolist()))
        fine_val_shape[0], fine_val_shape[2] = fine_val_shape[2], fine_val_shape[0]
        self.fine_val_shape = fine_val_shape
        val_shape = [1, c_dim, *fine_val_shape]
        fine_val = torch.zeros(val_shape).normal_(mean=0, std=0.0001)
        c[fine_key] = fine_val

        color_key = 'grid_color'
        color_val_shape = list(map(int, (xyz_len / color_grid_len).tolist()))
        color_val_shape[0], color_val_shape[2] = color_val_shape[2], color_val_shape[0]
        self.color_val_shape = color_val_shape
        val_shape = [1, c_dim, *color_val_shape]
        color_val = torch.zeros(val_shape).normal_(mean=0, std=0.01)
        c[color_key] = color_val

        self.shared_c = c

    def tracking(self, rank):
        """
        Tracking Thread.

        Args:
            rank (int): Thread ID.
        """

        # 这里就类似一定要进行了初始化、确定了世界坐标系，才能够进行Tracking；
        # 而NICE-SLAM这样的NeRF based SLAM初始化的办法就是把第一帧图像拍摄的相机位置作为世界坐标系原点，然后先建图再去跟踪；
        # 看NICE-SLAM论文的话就会发现，位姿都是优化出来的，而没有建图（没有训练一个网络出来）的话就也没办法优化出位姿。
        # should wait until the mapping of first frame is finished
        while (1):
            if self.mapping_first_frame[0] == 1:  #等待建图完成，位姿算出来
                break
            time.sleep(1)

        self.tracker.run()

    def mapping(self, rank):
        """
        Mapping Thread. (updates middle, fine, and color level)

        Args:
            rank (int): Thread ID.
        """

        self.mapper.run()

    def coarse_mapping(self, rank):
        """
        Coarse mapping Thread. (updates coarse level)

        Args:
            rank (int): Thread ID.
        """

        self.coarse_mapper.run()

    def run(self):
        """
        Dispatch Threads.
        """

        # 逐一启动
        processes = []
        for rank in range(3):
            if rank == 0:
                p = mp.Process(target=self.tracking, args=(rank,))
            elif rank == 1:
                p = mp.Process(target=self.mapping, args=(rank,))
            elif rank == 2:
                if self.coarse:
                    p = mp.Process(target=self.coarse_mapping, args=(rank,))
                else:
                    continue
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


# This part is required by torch.multiprocessing
if __name__ == '__main__':
    pass
