import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common import normalize_3d_coordinate

 #TODO 研一新生能力有限，这里有些开始看不懂了
class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    Modified based on the implementation of Gaussian Fourier feature mapping.
    基于高斯傅立叶特征建图的改进实现。
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    """

    def __init__(self, num_input_channels, mapping_size=93, scale=25, learnable=True):
        super().__init__()

        if learnable:
            self._B = nn.Parameter(torch.randn(
                (num_input_channels, mapping_size)) * scale)
        else:
            self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(x.dim())
        x = x @ self._B.to(x.device)  # 这里是矩阵相乘？
        return torch.sin(x)

"""
下面的函数不是太懂，我就让GPT来帮忙了
下面的Nerf_positional_embedding类是一个使用PyTorch实现的NeRF位置编码模块，
用于对3D空间中的点位置进行编码。该类包括了初始化方法`__init__`和前向传播方法`forward`。

在初始化方法中，该类接受`multires`和`log_sampling`两个参数，`multires`用于指定多分辨率的级别，
`log_sampling`用于指定是否使用对数采样。在前向传播方法中，首先对输入的x进行维度检查，
然后根据是否使用在均匀的频率区间内采样。接着对每个频率和周期函数进行计算，最后将所有结果拼接在一起并返回。

这个模块的作用是将输入的点位置进行位置编码，以便用于NeRF模型的训练和预测。
"""
class Nerf_positional_embedding(torch.nn.Module):
    """
    Nerf positional embedding.
    NeRF位置编码
    NeRF位置编码是指用于神经辅助表示(Neural Radiance Fields, NeRF)模型的一种位置编码方法。在NeRF模型中，
    位置编码用于将3D空间中的点位置映射到一个高维向量表示，以便模型可以对空间中的光线进行准确的预测和渲染。
    常见的位置编码方法包括使用多层感知机(MLP)对点的坐标进行编码，或者使用傅里叶特征进行编码。
    这些位置编码方法可以帮助NeRF模型学习3D空间中的光线密度和颜色信息，从而实现高质量的渲染效果。
    """

    def __init__(self, multires, log_sampling=True): #`log_sampling`用于指定是否在均匀的频率区间内采样
        super().__init__()
        self.log_sampling = log_sampling
        self.include_input = True
        self.periodic_fns = [torch.sin, torch.cos]
        self.max_freq_log2 = multires - 1
        self.num_freqs = multires
        self.max_freq = self.max_freq_log2
        self.N_freqs = self.num_freqs

    def forward(self, x):
        x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format( #维度检查
            x.dim())

        if self.log_sampling:
            freq_bands = 2. ** torch.linspace(0.,
                                              self.max_freq, steps=self.N_freqs)  #生成均匀的频率区间
        else:
            freq_bands = torch.linspace(
                2. ** 0., 2. ** self.max_freq, steps=self.N_freqs)
        output = []
        if self.include_input:
            output.append(x) #x是输入
        for freq in freq_bands:
            for p_fn in self.periodic_fns:  #对均匀的频率区间中的每个频率和周期函数进行计算
                output.append(p_fn(x * freq)) #算一遍sin(x*freq)，再算一遍cos(x*freq)
        ret = torch.cat(output, dim=1) #将所有结果拼接在一起并返回
        return ret


class DenseLayer(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    # TODO 重写reset_parameters是为了用指定的方法进行参数的初始化吗？为什么这么做？是xavier效果好吗？
    def reset_parameters(self) -> None:
        # xavier初始化方法中服从均匀分布U(−a,a) ，分布的参数a = gain * sqrt(6/fan_in+fan_out)
        # torch.nn.init.calculate_gain("relu")的值为1.4142135623730951，有什么深意？
        torch.nn.init.xavier_uniform_(
            self.weight, gain=torch.nn.init.calculate_gain(self.activation)) #xavier初始化
        if self.bias is not None:
            # 全部初始化为0
            torch.nn.init.zeros_(self.bias)


class Same(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.squeeze(0)
        return x


# 相关的链接
#  论文：https://arxiv.org/pdf/2003.04618.pdf
#  代码：https://github.com/autonomousvision/convolutional_occupancy_networks/blob/master/src/conv_onet/models/decoder.py
class MLP(nn.Module):
    """
    Decoder. Point coordinates not only used in sampling the feature grids, but also as MLP input.
    # 这个就是大名鼎鼎的decoder本尊,不仅用于在特征网格里采样，还用作MLP的输入
    Args:
        name (str): name of this decoder.
        dim (int): input dimension.
        c_dim (int): feature dimension.
        hidden_size (int): hidden size of Decoder network.
        n_blocks (int): number of layers.
        leaky (bool): whether to use leaky ReLUs.
        sample_mode (str): sampling feature strategy, bilinear|nearest.
        color (bool): whether or not to output color.
        skips (list): list of layers to have skip connections.
        grid_len (float): voxel length of its corresponding feature grid.
        pos_embedding_method (str): positional embedding method.
        concat_feature (bool): whether to get feature from middle level and concat to the current feature.
    """

    def __init__(self, name='', dim=3, c_dim=128,
                 hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear',
                 color=False, skips=[2], grid_len=0.16, pos_embedding_method='fourier', concat_feature=False):
        super().__init__()
        self.name = name
        self.color = color
        self.no_grad_feature = False
        self.c_dim = c_dim
        self.grid_len = grid_len
        self.concat_feature = concat_feature
        self.n_blocks = n_blocks
        self.skips = skips

        # TODO 这里为什么这么做？使用fc_c的目的是什么？为什么不用前面定义的DenseLayer？
        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        # 定义embedder，对位置进行编码
        if pos_embedding_method == 'fourier':
            embedding_size = 93
            self.embedder = GaussianFourierFeatureTransform( #高斯傅里叶特征变换
                dim, mapping_size=embedding_size, scale=25)
        elif pos_embedding_method == 'same':
            embedding_size = 3
            self.embedder = Same()
        elif pos_embedding_method == 'nerf': #NeRF位置编码
            if 'color' in name:  #如果包含颜色，则在均匀的频率区间内采样
                multires = 10 #分辨率设为10？
                self.embedder = Nerf_positional_embedding(
                    multires, log_sampling=True)
            else:
                multires = 5
                self.embedder = Nerf_positional_embedding(
                    multires, log_sampling=False)
            embedding_size = multires * 6 + 3
        elif pos_embedding_method == 'fc_relu':
            embedding_size = 93
            self.embedder = DenseLayer(dim, embedding_size, activation='relu')

        # 参考nerf-pytorch实现的话，这里才是论文所描述的网络结构，但前面的fc_c又是什么呢？
        self.pts_linears = nn.ModuleList(
            [DenseLayer(embedding_size, hidden_size, activation="relu")] +
            [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips  # 初始化一下
             else DenseLayer(hidden_size + embedding_size, hidden_size, activation="relu") for i in
             range(n_blocks - 1)])

        # 输出层，去看NICE类的forward函数就能明白为什么这里color输出的维度是4
        if self.color:
            self.output_linear = DenseLayer(
                hidden_size, 4, activation="linear")
        else:
            self.output_linear = DenseLayer(
                hidden_size, 1, activation="linear")

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

    def sample_grid_feature(self, p, c):   # 在特征网格内采样
        p_nor = normalize_3d_coordinate(p.clone(), self.bound)
        p_nor = p_nor.unsqueeze(0)
        vgrid = p_nor[:, :, None, None].float()

        # 下面这个函数可以参考：https://zhuanlan.zhihu.com/p/112030273 和 https://zhuanlan.zhihu.com/p/495758460
        # 官方文档：https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True,
                          mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, c_grid=None):
        # 先提取每个点的特征
        if self.c_dim != 0:
            c = self.sample_grid_feature(  #在特征网格内采样
                p, c_grid['grid_' + self.name]).transpose(1, 2).squeeze(0)

            if self.concat_feature:
                # only happen to fine decoder, get feature from middle level and concat to the current feature
                # 如果在fine层，则采样后和middle层拼接
                with torch.no_grad():
                    c_middle = self.sample_grid_feature(
                        p, c_grid['grid_middle']).transpose(1, 2).squeeze(0)
                c = torch.cat([c, c_middle], dim=1)

        # 对位置进行编码
        p = p.float()
        embedded_pts = self.embedder(p)
        h = embedded_pts

        # TODO 不太理解下面的做法
        #  这里的意思是，位置编码后的h先通过pts_linears，然后再加上fc_c处理过的feature？
        #  去简单看了一下ConvONet的代码，他们的代码里面就有类似这样的处理，但是没有get到为什么要这样做
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)  #对每一层加一个relu
            if self.c_dim != 0:
                h = h + self.fc_c[i](c)  #再加上fc_c处理过的feature？
            if i in self.skips:
                h = torch.cat([embedded_pts, h], -1)  #与编码后的点进行拼接

        # 输出层，输出的维度和前面的定义有关
        out = self.output_linear(h)
        if not self.color:
            out = out.squeeze(-1)
        return out


class MLP_no_xyz(nn.Module): #coarse-level不需要学习高频的细节，所以coarse-level使用MLP_no_xyz
    """
    Decoder. Point coordinates only used in sampling the feature grids, not as MLP input.
    # decoder，只用作在特征网格内采样，没有MLP的输入
    Args:
        name (str): name of this decoder.
        dim (int): input dimension.
        c_dim (int): feature dimension.
        hidden_size (int): hidden size of Decoder network.
        n_blocks (int): number of layers.
        leaky (bool): whether to use leaky ReLUs.
        sample_mode (str): sampling feature strategy, bilinear|nearest.
        color (bool): whether or not to output color.
        skips (list): list of layers to have skip connection.
        grid_len (float): voxel length of its corresponding feature grid.
    """

    # 上面的这个注释看的不太明白，论文里面说coarse-level不需要高斯位置编码，但是Pipeline还是画了coarse level的decoder接受了xyz输入？
    def __init__(self, name='', dim=3, c_dim=128,
                 hidden_size=256, n_blocks=5, leaky=False,
                 sample_mode='bilinear', color=False, skips=[2], grid_len=0.16):
        super().__init__()
        self.name = name
        self.no_grad_feature = False  # 未使用
        self.color = color
        self.grid_len = grid_len
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.skips = skips

        # TODO 第一层的输入就是hidden_size？不应该是c_dim？？？
        self.pts_linears = nn.ModuleList(
            [DenseLayer(hidden_size, hidden_size, activation="relu")] +
            [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips
             else DenseLayer(hidden_size + c_dim, hidden_size, activation="relu") for i in range(n_blocks - 1)])

        if self.color:
            self.output_linear = DenseLayer(
                hidden_size, 4, activation="linear")
        else:
            self.output_linear = DenseLayer(
                hidden_size, 1, activation="linear")

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

    def sample_grid_feature(self, p, grid_feature):
        p_nor = normalize_3d_coordinate(p.clone(), self.bound)
        p_nor = p_nor.unsqueeze(0)
        vgrid = p_nor[:, :, None, None].float()
        c = F.grid_sample(grid_feature, vgrid, padding_mode='border',
                          align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, c_grid, **kwargs):
        c = self.sample_grid_feature(
            p, c_grid['grid_' + self.name]).transpose(1, 2).squeeze(0)
        h = c
        # 这里就没有位置编码了，并且pts_linears直接处理grid feature
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([c, h], -1)
        out = self.output_linear(h)
        if not self.color:
            out = out.squeeze(-1)
        return out


class NICE(nn.Module):
    """    
    Neural Implicit Scalable Encoding.

    Args:
        dim (int): input dimension.
        c_dim (int): feature dimension.
        coarse_grid_len (float): voxel length in coarse grid.
        middle_grid_len (float): voxel length in middle grid.
        fine_grid_len (float): voxel length in fine grid.
        color_grid_len (float): voxel length in color grid.
        hidden_size (int): hidden size of decoder network
        coarse (bool): whether or not to use coarse level.
        pos_embedding_method (str): positional embedding method.
    """

    def __init__(self, dim=3, c_dim=32,
                 coarse_grid_len=2.0, middle_grid_len=0.16, fine_grid_len=0.16,
                 color_grid_len=0.16, hidden_size=32, coarse=False, pos_embedding_method='fourier'):
        super().__init__()

        # coarse-level不需要学习高频的细节，所以这里使用MLP_no_xyz
        if coarse:
            self.coarse_decoder = MLP_no_xyz(
                name='coarse', dim=dim, c_dim=c_dim, color=False, hidden_size=hidden_size, grid_len=coarse_grid_len)

        self.middle_decoder = MLP(name='middle', dim=dim, c_dim=c_dim, color=False,
                                  skips=[2], n_blocks=5, hidden_size=hidden_size,
                                  grid_len=middle_grid_len, pos_embedding_method=pos_embedding_method)
        self.fine_decoder = MLP(name='fine', dim=dim, c_dim=c_dim * 2, color=False,
                                skips=[2], n_blocks=5, hidden_size=hidden_size,
                                grid_len=fine_grid_len, concat_feature=True, pos_embedding_method=pos_embedding_method)
        self.color_decoder = MLP(name='color', dim=dim, c_dim=c_dim, color=True,
                                 skips=[2], n_blocks=5, hidden_size=hidden_size,
                                 grid_len=color_grid_len, pos_embedding_method=pos_embedding_method)

    def forward(self, p, c_grid, stage='middle', **kwargs):
        """
            Output occupancy/color in different stage.
        """
        # 这里的stage应该是按照优化的过程来的，Mapping线程三个阶段的优化，以及Coarse level的单独优化，差不多能对应到下面这几个stage
        # coarse和middle的过程是一样的，经过decoder得到occupancy
        # fine stage这里经过fine_decoder得到残差fine_occ，再经过middle_decoder得到occupancy值middle_occ，
        # 最后返回的是论文Pipeline描述的“Fine-level Occupancy”
        # 这里除了color stage之外，所有的raw都赋值了torch.zeros(xxx_occ.shape[0], 4).to(device).float()
        # 这里raw被初始化为shape=torch.Size([xxx_occ.shape[0], 4])的tensor
        # 我认为raw的每一行表示的应该是[R,G,B,occupancy]，这样做是为了统一格式
        device = f'cuda:{p.get_device()}'
        if stage == 'coarse':
            occ = self.coarse_decoder(p, c_grid)
            occ = occ.squeeze(0)
            raw = torch.zeros(occ.shape[0], 4).to(device).float()
            raw[..., -1] = occ
            return raw
        elif stage == 'middle':
            middle_occ = self.middle_decoder(p, c_grid)
            middle_occ = middle_occ.squeeze(0)
            raw = torch.zeros(middle_occ.shape[0], 4).to(device).float()
            raw[..., -1] = middle_occ
            return raw
        elif stage == 'fine':
            fine_occ = self.fine_decoder(p, c_grid)
            raw = torch.zeros(fine_occ.shape[0], 4).to(device).float()
            middle_occ = self.middle_decoder(p, c_grid)
            middle_occ = middle_occ.squeeze(0)
            raw[..., -1] = fine_occ + middle_occ
            return raw
        elif stage == 'color':
            fine_occ = self.fine_decoder(p, c_grid)
            raw = self.color_decoder(p, c_grid)
            middle_occ = self.middle_decoder(p, c_grid)
            middle_occ = middle_occ.squeeze(0)
            raw[..., -1] = fine_occ + middle_occ
            return raw
