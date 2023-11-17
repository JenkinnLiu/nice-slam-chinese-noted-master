import argparse
import random

import numpy as np
import torch

from src import config
from src.NICE_SLAM import NICE_SLAM


def setup_seed(seed):
    """
    设置固定的随机数种子，保证代码多次运行的效果尽可能相同（但是有可能会让性能降低）
    参考：https://pytorch.org/docs/stable/notes/randomness.html
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 将这个 flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法。
    # 如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的
    torch.backends.cudnn.deterministic = True


def main():
    # setup_seed(20)

    # =============== 参数解析 ===============
    # 先解析配置文件，在解析命令行参数，命令行参数有更高的优先级
    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    # 创建一个互斥组，保证nice和imap只能二选一，required=False说明命令行参数可以不包含是nice还是imap(后面设置了默认nice=True)
    nice_parser = parser.add_mutually_exclusive_group(required=False)
    nice_parser.add_argument('--nice', dest='nice', action='store_true')
    nice_parser.add_argument('--imap', dest='nice', action='store_false')
    parser.set_defaults(nice=True)
    # 开始解析参数，先解析命令行参数，再解析配置文件的参数
    args = parser.parse_args()
    cfg = config.load_config(
        args.config, 'configs/nice_slam.yaml' if args.nice else 'configs/imap.yaml')

    # =============== 初始化 ===============
    slam = NICE_SLAM(cfg, args)

    # =============== 运行 ===============
    slam.run()


if __name__ == '__main__':
    main()
