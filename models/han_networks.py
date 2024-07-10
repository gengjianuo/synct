import os

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import math
import numpy as np
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from . import residual_transformers
from math import log


###############################################################################
# Helper Functions
###############################################################################

class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, threshold=0.001, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)

            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm3d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


# def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
#     """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
#     Parameters:
#         net (network)      -- the network to be initialized
#         init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
#         gain (float)       -- scaling factor for normal, xavier and orthogonal.
#         gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
#
#     Return an initialized network.
#     """
#     if len(gpu_ids) > 0:
#         assert(torch.cuda.is_available())
#         net.to(gpu_ids[0])
#         net = torch.nn.DataParallel(net, gpu_ids)
#     init_weights(net, init_type, init_gain=init_gain)
#     return net

def init_net_G(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    ######################################3
        init_weights(net, init_type, init_gain=init_gain)
    ##################################
    return net

def init_net_D(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    ######################################
    #weights_path='/tmp/pycharm_project_838/src/checkpoints/experiment_name/best_net_D.pth'
    #if os.path.exists(weights_path):
    #    weights_dict = torch.load(weights_path)
    #    load_weights_dict={k:v for k,v in weights_dict.items()}
    #    net.load_state_dict(load_weights_dict,strict=False)
    #else:
        init_weights(net, init_type, init_gain=init_gain)
    ##################################
    return net



def define_G(input_nc, output_nc, ngf, netG, norm='instance', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[] ):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'resnet_9blocks':
        net = ResnetGenerator21(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 5, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'local':
        net = LocalEnhancer(input_nc, output_nc, ngf=64, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.InstanceNorm3d, padding_type='reflect')
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net_G(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net_D(net, init_type, init_gain, gpu_ids)

##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

#        print(f"Current GPU Index:{current_device}")
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            # print(prediction)
            # print(target_tensor)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp
        return gradient_penalty, gradients
    else:
        return 0.0, None
def quantile_loss(y_true, y_pred, q):
    error = y_true - y_pred
    # quantile_loss = np.mean(np.where(error >= 0, q * error, (q - 1) * error))
    quantile_loss = torch.mean(torch.where(error >= 0, q * error, (q - 1) * error))
    return quantile_loss



class Bottle2neckXse(nn.Module):
    expansion = 4

    def __init__(self, inplanes =256, planes = 256, baseWidth = 4, cardinality=8, stride=1, downsample = True, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
            scale: number of scale.
            type: 'normal': normal set. 'stage': frist blokc of a new stage.
        """
        super(Bottle2neckXse, self).__init__()

        D = int(math.floor(planes * (baseWidth/64.0)))
        C = cardinality

        self.conv1 = nn.Conv3d(inplanes, D*C*scale, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.InstanceNorm3d(D*C*scale)

        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool3d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv3d(D*C, D*C, kernel_size=3, stride = stride, padding=1, groups=C, bias=False))
          bns.append(nn.InstanceNorm3d(D*C))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv3d(D*C*scale, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.InstanceNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Conv3d(512,256,kernel_size=1, stride=1, padding=0, bias=False)
        self.width = D*C
        self.stype = stype
        self.scale = scale
        self.se1 = SELayer(256)
        self.se2 = SELayer(384)
        self.se3 = SELayer(512)
        # self.nam = NAM(256)
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
            if out.shape[1] == 256:
                out = self.se1(out)
            else:
                out = self.se2(out)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
          out = self.se3(out)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)
          out = self.se3(out)
        # out = self.conv3(out)
        # out = self.bn3(out)

        if self.downsample is not None:
            out = self.downsample(out)

        out += residual
        out = self.relu(out)
        # z = self.nam.forward(out)
        # y = z * out
        return out

class LSKmodule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        self.convl = nn.Conv3d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv0_s = nn.Conv3d(dim, dim // 2, 1)
        self.conv1_s = nn.Conv3d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv3d(2, 2, 7, padding=3)
        self.conv_m = nn.Conv3d(dim // 2, dim, 1)
    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.convl(attn1)
        attn1 = self.conv0_s(attn1)
        attn2 = self.conv1_s(attn2)
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        # attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = attn1 * sig[:, 0, :, :, :] + attn2 * sig[:, 1, :, :, :]
        attn = self.conv_m(attn)
        return x * attn


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.InstanceNorm3d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x
class SAPblock2(nn.Module):
    def __init__(self, in_channels):
        super(SAPblock2, self).__init__()
        self.conv3x3 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, dilation=1, kernel_size=3,
                                 padding=1)
        # self.conv3x32 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, dilation=2, kernel_size=3,
        #                          padding=2)
        # self.conv3x33 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, dilation=4, kernel_size=3,
        #                   padding=4)
        self.bn = nn.ModuleList([nn.InstanceNorm3d(in_channels), nn.InstanceNorm3d(in_channels), nn.InstanceNorm3d(in_channels) , nn.InstanceNorm3d(in_channels//2), nn.InstanceNorm3d(2),nn.InstanceNorm3d(1)])
        self.conv1x1 = nn.ModuleList(
            [nn.Conv3d(in_channels=2 * in_channels, out_channels=in_channels, dilation=1, kernel_size=1, padding=0),
             nn.Conv3d(in_channels=2 * in_channels, out_channels=in_channels, dilation=1, kernel_size=1, padding=0)])
        self.conv3x3_1 = nn.ModuleList(
            [nn.Conv3d(in_channels=in_channels, out_channels=in_channels // 2, dilation=1, kernel_size=3, padding=1),
             nn.Conv3d(in_channels=in_channels, out_channels=in_channels // 2, dilation=1, kernel_size=3, padding=1)])
        self.conv3x3_2 = nn.ModuleList(
            [nn.Conv3d(in_channels=in_channels // 2, out_channels=2, dilation=1, kernel_size=3, padding=1),
             nn.Conv3d(in_channels=in_channels // 2, out_channels=1, dilation=1, kernel_size=3, padding=1)])
        self.conv_last = ConvBnRelu(in_planes=in_channels, out_planes=in_channels, ksize=1, stride=1, pad=0, dilation=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_size = x.size()

        branches_1 = self.conv3x3(x)
        branches_1 = self.relu(self.bn[0](branches_1))

        branches_2 = F.conv3d(x, self.conv3x3.weight, padding=2, dilation=2)  # share weight
        branches_2 = self.relu(self.bn[1](branches_2))

        branches_3 = F.conv3d(x, self.conv3x3.weight, padding=4, dilation=4)  # share weight
        branches_3 = self.relu(self.bn[2](branches_3))

        feat = torch.cat([branches_1, branches_2], dim=1)
        # feat=feat_cat.detach()
        feat = self.relu(self.bn[2](self.conv1x1[0](feat)))
        feat = self.relu(self.bn[3](self.conv3x3_1[0](feat)))
        att = self.bn[4](self.conv3x3_2[0](feat))
        att = F.softmax(att, dim=1)

        att_1 = att[:, 0, :,:, :].unsqueeze(1)
        att_2 = att[:, 1, :,:, :].unsqueeze(1)

        fusion_1_2 = att_1 * branches_1 + att_2 * branches_2

        feat1 = torch.cat([fusion_1_2, branches_3], dim=1)
        # feat=feat_cat.detach()
        feat1 = self.relu(self.bn[2](self.conv1x1[0](feat1)))
        feat1 = self.relu(self.bn[3](self.conv3x3_1[0](feat1)))
        att1 = self.bn[5](self.conv3x3_2[1](feat1))
        att1 = F.softmax(att1, dim=1)

        # att1 = self.bi(att1)
        ax = self.relu(att1 * x + x)
        ax = self.conv_last(ax)

        return ax,att1

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _,_ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class ResnetGenerator21(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    #sap2l+9res2net+unet(csa+multi)
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, norm_layer=nn.InstanceNorm3d, n_blocks=9):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGenerator21, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d
        self.pad = nn.Sequential(nn.ReplicationPad3d(3))
        self.conv7 = nn.Sequential(nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                                   norm_layer(ngf),
                                   nn.ReLU(True))
        self.conv3_1 = nn.Sequential(nn.Conv3d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                                     norm_layer(ngf * 2),
                                     nn.ReLU(True))
        self.conv3_2 = nn.Sequential(nn.Conv3d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias),
                                     norm_layer(ngf * 4),
                                     nn.ReLU(True))
        # self.resnet = self.r(ngf=64, norm_layer=nn.InstanceNorm3d, use_dropout=False, n_blocks=9,
        #                      padding_type='reflect')
        self.res1 = Bottle2neckXse(inplanes=256, planes=256, baseWidth=4, cardinality=8, stride=1)
        self.csa1 = Bottleneckcsa1(64, 16)
        self.csa2 = Bottleneckcsa1(128, 32)
        # self.multi1 = MultiScaleDilatedResidualBlock(64, 64)
        # self.multi2 = MultiScaleDilatedResidualBlock(128, 128)
        self.upconv1 = nn.Sequential(nn.ConvTranspose3d(ngf * 4, ngf * 2,
                                                        kernel_size=3, stride=2,
                                                        padding=1, output_padding=1,
                                                        bias=use_bias),
                                     norm_layer(ngf * 2),
                                     nn.ReLU(True))
        self.upconv2 = nn.Sequential(nn.ConvTranspose3d(ngf * 2, ngf,
                                                        kernel_size=3, stride=2,
                                                        padding=1, output_padding=1,
                                                        bias=use_bias),
                                     norm_layer(ngf),
                                     nn.ReLU(True))
        self.conv3_3 = nn.Sequential(nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0))
        # self.conv1 = nn.Sequential(nn.Conv3d(ngf * 8, ngf * 4, kernel_size=1, stride=1, padding=0),
        #                            nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv3d(ngf * 4, ngf * 2, kernel_size=1, stride=1, padding=0),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv3d(ngf * 2, ngf, kernel_size=1, stride=1, padding=0),
                                   nn.ReLU())
        # self.conv5 = nn.Sequential(nn.Conv3d(ngf + 1, ngf, kernel_size=3, stride=1, padding=1),
        #                            nn.ReLU())
        # self.conv1_1 = nn.Sequential(nn.Conv3d(ngf * 4, ngf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
        #                              norm_layer(ngf * 2),
        #                              nn.ReLU(True))
        self.jihuo = nn.Sequential(nn.Tanh())


        # self.gab = group_aggregation_bridge(dim_xh=128, dim_xl=128, k_size=3, d_list=[1, 2, 5, 7])
        # self.gt_conv2 = nn.Sequential(nn.Conv3d(128, 1, 1), nn.InstanceNorm3d(1), nn.ReLU(True))
        self.sap1 = SAPblock2(256)
        # self.ema1 = EMA(64)
        # self.ema2 = EMA(128)
        # self.multi4 = MultiScaleDilatedResidualBlock4(in_channels=64, out_channels=128, dilation_rates=[1, 2, 4])
        # self.ghpa1 = GHPA(128, 128, x=10, y=10, z=2)
        # self.ghpa2 = GHPA(64, 64, x=10, y=10, z=2)
        # self.ghpa3 = GHPA(256, 256, x=10, y=10, z=2)
    # def r(self, ngf=64, norm_layer=nn.InstanceNorm3d, use_dropout=False, n_blocks=9,
    #       padding_type='reflect'):
    #     n_downsampling = 2
    #     mult = 2 ** n_downsampling
    #     model = []
    #     if type(norm_layer) == functools.partial:
    #         use_bias = norm_layer.func == nn.InstanceNorm3d
    #     else:
    #         use_bias = norm_layer == nn.InstanceNorm3d
    #     for i in range(n_blocks):
    #         model += [
    #             ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
    #                         use_bias=use_bias)]
    #     return nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        out1 = self.pad(input)
        out2 = self.conv7(out1)
        out3 = self.conv3_1(out2)
        # out5 = self.multi4(out2)
        # out3 = torch.cat((out3,out5),1)
        # out3 = self.conv1_1(out3)
        # out3 = out5+out3
        # del out5
        out4 = self.conv3_2(out3)
        out1_1 = out4
        # out2 = self.multi1(out2)
        # out3 = self.multi2(out3)
        out2 = self.csa1(out2)
        out3 = self.csa2(out3)
        out1_1, a1 = self.sap1(out1_1)
        a1 = F.interpolate(a1, scale_factor=4, mode='trilinear',
                           align_corners=True)
        for i in range(9):
            out1_1 = self.res1(out1_1)
        # out2_1 = self.upconv1(self.conv1(torch.cat((out4, out1_1), 1)))
        out2_1 = self.upconv1(out1_1)
        out3_1 = self.upconv2(self.conv2(torch.cat((out2_1, out3), 1)))
        out4_1 = self.pad(self.conv4(torch.cat((out3_1, out2), 1)))
        out = self.jihuo(self.conv3_3(out4_1))
        return out, a1


class Bottleneckcsa1(nn.Module):
    """ResNet Bottleneck
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 radix=2, cardinality=2, bottleneck_width=64,
                 dilation=1,rectified_conv=False, expansion = 4,rectify_avg=False,
                 norm_layer=nn.InstanceNorm3d, dropblock_prob=0.0):
        super(Bottleneckcsa1, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.inplanes = inplanes
        self.maxpool = nn.MaxPool3d(kernel_size=2)
        down_layers = []
        down_layers.append(nn.Conv3d(self.inplanes, planes * expansion,
                                     kernel_size=1, stride=1, bias=False))
        down_layers.append(norm_layer(planes * expansion))
        downsample = nn.Sequential(*down_layers)
        self.conv1 = nn.Conv3d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.avd_layer = nn.AvgPool3d(3, stride, padding=1)

        self.conv2 = SplAtConv3d1(
            group_width, group_width, kernel_size=3,
            stride=stride, padding=dilation,
            dilation=dilation, groups=cardinality, bias=False,
            radix=radix,
            norm_layer=norm_layer)

        self.conv3 = nn.Conv3d(
            group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        # self.dp = nn.Dropout(0.1)

    def forward(self, x):
        # x = self.maxpool(x)
        residual = x

        out = self.conv1(x)
        # out = self.dp(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.dp(out)

        # out = self.avd_layer(out)

        out = self.conv3(out)
        # out = self.dp(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class SplAtConv3d1(nn.Module):
    """Split-Attention Conv3d
    """

    def __init__(self, in_channels, channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True,
                 radix=2, reduction_factor=4,
                  norm_layer=nn.InstanceNorm3d,
                  **kwargs):
        super(SplAtConv3d1, self).__init__()
        # padding = _pair(padding)
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, dilation,
                           groups=groups * radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels * radix)
            self.bn2 = norm_layer(channels)
        self.relu = nn.ReLU(inplace=True)
        self.lsk = LSKmodule(in_channels)
    def forward(self, x):

        x = self.conv(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.lsk(x)
        return x.contiguous()


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm3d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm3d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        self.net = [
            nn.Conv3d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True)]
        self.net1 = [nn.Conv3d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)
            ]

        self.net = nn.Sequential(*self.net)
        self.net1 = nn.Sequential(*self.net1)
        # self.ema = EMA(64)
        # self.se = SELayer(64)
        # self.non = NonLocalBlock(64)
    def forward(self, input):
        """Standard forward."""
        out = self.net(input)
        # out = self.ema(out)
        # out = self.se(out)
        # out = self.non(out)
        out = self.net1(out)
        return out
