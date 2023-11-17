import os

import torch


class Logger(object):
    """
    Save checkpoints to file.
    保存checkpoints到文件中，相当于log日志
    """

    def __init__(self, cfg, args, slam
                 ):
        self.verbose = slam.verbose #至今不知道这个verbose是个啥
        self.ckptsdir = slam.ckptsdir #checkpointsDIR,用于保存checkpoints的文件
        self.shared_c = slam.shared_c #共享特征网格shared_c数组
        self.gt_c2w_list = slam.gt_c2w_list #真实位姿的list
        self.shared_decoders = slam.shared_decoders  #共享decoders
        self.estimate_c2w_list = slam.estimate_c2w_list #估计位姿的list

    def log(self, idx, keyframe_dict, keyframe_list, selected_keyframes=None):
        path = os.path.join(self.ckptsdir, '{:05d}.tar'.format(idx)) #log日志
        torch.save({ #保存以下参数至文件
            'c': self.shared_c,  #c数组
            'decoder_state_dict': self.shared_decoders.state_dict(), #decoder的状态参数
            'gt_c2w_list': self.gt_c2w_list, #真实位姿的list
            'estimate_c2w_list': self.estimate_c2w_list,  #估计位姿的list
            'keyframe_list': keyframe_list, #关键帧的list
            # 'keyframe_dict': keyframe_dict, # to save keyframe_dict into ckpt, uncomment this line
            'selected_keyframes': selected_keyframes, #选择的关键帧
            'idx': idx, #帧id
        }, path, _use_new_zipfile_serialization=False)

        if self.verbose:
            print('Saved checkpoints at', path)
