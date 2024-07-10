import torch
from .base_model import BaseModel
from . import han_networks
from .edge_loss import cal_gradient_loss,spacialGradient_3d,spacialGradient_3d1
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile
import torch.distributed as dist

class HanPix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        # parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_Edge', type=float, default=10.0, help='weight for edge loss')
            parser.add_argument('--lambda_ql', type=float, default=10.0, help='weight for edge loss')
            parser.add_argument('--lambda_GAN', type=float, default=5.0, help='weight for edge loss')
            parser.add_argument('--lambda_D', type=float, default=5.0, help='weight for edge loss')
            parser.add_argument('--lambda_L2', type=float, default=1.0, help='weight for edge loss')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'G_Edge', 'ql', 'D_real', 'D_fake' ]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B' ]
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        if self.isTrain:
            self.training_mode = opt.training_mode
        self.netG = han_networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                          not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            if self.training_mode == 'GAN':  # concurrent work: unconditional discriminator
                self.netD = han_networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, 
                    opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            else:
                self.netD = han_networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                    opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            
        if self.isTrain:
            # define loss functions
            self.criterionGAN = han_networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # self.creation = han_networks.CharbonnierLoss(epsilon=1e-3).to(self.device)

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            ##################################################

            #################################################################

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'

        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
    #######################################################################

    ##################################################33
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        #current_device = torch.cuda.current_device()
#         textgpu = []
        # for name in self.optimizer_G.param_groups[0]["params"]:
        #     print(f"Device{name.device}")
        # for name, param in self.optimizer_D.param_groups[0]["params"]:
        #     print(f"Parameter{name}:Device{param.device}")
        #device=self.real_A.device
        # print('device')
        self.fake_B, self.a1 = self.netG(self.real_A) # G(A)
        # self.fake_B= self.netG(self.real_A)
         # self.fake_B, self.a1,self.a2,self.a3 = self.netG(self.real_A)
    def backward_D(self, train_lossD):
        """Calculate GAN loss for the discriminator"""

        # Fake; stop backprop to the generator by detaching fake_B
        if self.training_mode == 'GAN':
            fake_AB = self.fake_B
        else: # we use conditional GANs; we need to feed both input and output to the discriminator
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)

        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        if self.training_mode == 'GAN':
            real_AB = self.real_B
        else:
            real_AB = torch.cat((self.real_A, self.real_B), 1)
    
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # real1 = torch.sigmoid(pred_real)
        # self.loss_dice_loss_real = han_networks.dice_loss(1, real1).to(self.device)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        train_lossD.update(self.loss_D)
        if self.training_mode != 'unet':
            self.loss_D.backward()
        ###################################################

         ###############################################################################


    def backward_G(self, train_lossG):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        if self.training_mode == 'GAN':
            fake_AB = self.fake_B          
        else:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
                 
        pred_fake = self.netD(fake_AB)
        if self.training_mode == 'unet':
            self.loss_G_GAN = 0
        else:
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # self.loss_ql = han_networks.quantile_loss(self.real_B,self.fake_Ba,0.5) * self.opt.lambda_ql

        # Second, G(A) = B
        self.tidu = spacialGradient_3d1(self.real_B)
        self.tidu1 = spacialGradient_3d1(self.a1)
        # 计算最小值和最大值
        self.min_value = torch.min(self.tidu)
        self.max_value = torch.max(self.tidu)
        self.min_value1 = torch.min(self.tidu1)
        self.max_value1 = torch.max(self.tidu1)
        # # 归一化到 (0, 1) 区间
        self.normalized_tidu = (self.tidu - self.min_value) / (self.max_value - self.min_value)
        self.normalized_tidu1 = (self.tidu1 - self.min_value1) / (self.max_value1 - self.min_value1)
        # self.normalized_tidu = torch.exp(self.normalized_tidu)

        self.loss_ql = han_networks.quantile_loss(self.normalized_tidu, self.normalized_tidu1, 0.5) * self.opt.lambda_ql
        # self.loss_ql2 = han_networks.quantile_loss(self.normalized_tidu, self.a2, 0.5) * self.opt.lambda_ql
        # self.loss_ql3 = han_networks.quantile_loss(self.normalized_tidu, self.a3, 0.5) * self.opt.lambda_ql
        # self.loss_ql_a1 = han_networks.quantile_loss(self.real_B, self.a1, 0.5) * self.opt.lambda_ql
        # self.loss_ql_a2 = han_networks.quantile_loss(self.real_B, self.a2, 0.5) * self.opt.lambda_ql
        # self.loss_ql_a3 = han_networks.quantile_loss(self.real_B, self.a3, 0.5) * self.opt.lambda_ql
        # self.result = 0 > self.real_B > -0.5
        # self.new_tensor = torch.ones_like(self.real_B)
        # self.new_tensor[self.result] = 3
        # self.normalized_tidu = torch.softmax(self.normalized_tidu, dim=1)
        # self.loss_L2 = han_networks.l2_loss(self.new_tensor * self.real_B,
        #                                     self.new_tensor * self.b1) * self.opt.lambda_L2
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G_Edge = cal_gradient_loss(self.fake_B, self.real_B) * self.opt.lambda_Edge
        # fake1 = torch.sigmoid(self.fake_B)
        # self.loss_dice_loss_real = han_networks.dice_loss(self.real_B, fake1).to(self.device)
        # self.loss_creation = self.creation(self.fake_B-self.real_B)
        # combine loss and calculate gradients
        # self.loss_G = self.loss_G_GAN + self.loss_G_L1  + self.loss_G_Edge
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_Edge + self.loss_ql
        train_lossG.update(self.loss_G)
        self.loss_G.backward()
        ###################################################
        ###############################################################################


    def optimize_parameters(self, train_lossG, train_lossD):
        self.forward()  # compute fake images: G(A)
        # update D
        if self.training_mode == 'unet':
            self.set_requires_grad(self.netD, False)
        else:
            self.set_requires_grad(self.netD, True)
        
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D(train_lossD)  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights

        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G(train_lossG)  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
    def val_forward(self):
        self.forward()
        self.val_loss = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        return self.val_loss

