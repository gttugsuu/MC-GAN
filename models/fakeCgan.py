import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from scipy import misc
import random

class fakecgan(nn.Module):
    def __init__(self, opt):
        super(fakecgan, self).__init__()

        self.netG_3d = networks.define_G_3d(opt.input_nc, opt.input_nc, 
                                            norm=opt.norm, groups=opt.grps, 
                                            gpu_ids=self.gpu_ids)

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                    'resnet_6blocks', 'batch', False, 
                                    gpu_ids=self.gpu_ids)

        self.load_network(self.netG_3d, 'G_3d', opt.which_epoch)
        self.load_network(self.netG, 'G', opt.which_epoch)

    def set_input(self, input):
        input_A = input['A']
        input_B = input['B']        
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths']

    def test(self):
        # few shot input
        self.real_A = self.input_A
        # fake output
        self.real_A_indep = self.netG_3d.forward(self.real_A.unsqueeze(2))
        self.fake_B = self.netG.forward(self.real_A_indep.squeeze(2))
        # real output
        self.real_B = self.input_B

    #get image paths
    def get_image_paths(self):
        return self.image_paths

