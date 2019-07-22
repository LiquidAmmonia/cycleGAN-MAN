import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools

import torch.nn as nn
from torchvision import models, transforms
import sys
import argparse
from visualizer import Visualizer
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from data.data_loader import CreateDataLoader
import time
import PIL.Image
from net import networks as nets
import util
from util import AvgMeter, ImagePool

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='',
                         help='path to images (should have subfolders trainA, trainB, valA, valB, testA, testB)')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks', help='selects model to use for netG')
parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data')
parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
parser.add_argument('--serial_batches', action='store_true',
                         help='if true, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
parser.add_argument('--train_display_id', type=int, default=1, help='window id of the web display')
parser.add_argument('--val_display_id', type=int, default=10, help='window id of the web display')
parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
parser.add_argument('--display_single_pane_ncols', type=int, default=4,
                         help='if positive, display all images in a single visdom web panel with certain number of images per row.')
parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                         help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop',
                         help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width]')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
parser.add_argument('--no_flip', action='store_true', help='use dropout for the generator')
parser.add_argument('--resume', default = '')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=300,
                         help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--print_iter', type=int, default=50, help='frequency of showing training results on console')
parser.add_argument('--display_iter', type=int, default=50, help='frequency of showing training results on console')
parser.add_argument('--save_iter', type=int, default=100, help='frequency of showing training results on console')
parser.add_argument('--ckpt_path', default = '')

opt = parser.parse_args()
print(opt)

train_visual = Visualizer(opt.train_display_id, 'train', 5)
val_visual = Visualizer(opt.val_display_id, 'val', 5)
if not os.path.exists(opt.ckpt_path):
    os.makedirs(opt.ckpt_path)

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)

opt.pahse = 'val'
val_data_loader = CreateDataLoader(opt)
val_dataset = val_data_loader.load_data()

## define models
#1---256x256 stage
#2---128x128 stage
#3---64x64 stage

GA = nets.define_G(input_nc=3,output_nc=3,ngf=64,
                    which_model_netG='resnet_9blocks', gpu_ids=[0],init_type='normal')
GB = nets.define_G(input_nc=3,output_nc=3,ngf=64,
                    which_model_netG='resnet_9blocks', gpu_ids=[0], init_type='normal')
DA1 = nets.define_D(input_nc=2* 3, ndf=64,
                    which_model_netD='n_layers',gpu_ids=[0],init_type='normal', n_layers_D=1)
DA2 = nets.define_D(input_nc=2* 3, ndf=64,
                    which_model_netD='n_layers',gpu_ids=[0],init_type='normal', n_layers_D=1)
DA3 = nets.define_D(input_nc=2* 3, ndf=64,
                    which_model_netD='n_layers',gpu_ids=[0],init_type='normal', n_layers_D=1)
DB1 = nets.define_D(input_nc=2* 3, ndf=64,
                    which_model_netD='n_layers',gpu_ids=[0],init_type='normal', n_layers_D=1)
DB2 = nets.define_D(input_nc=2* 3, ndf=64,
                    which_model_netD='n_layers',gpu_ids=[0],init_type='normal', n_layers_D=1)
DB3 = nets.define_D(input_nc=2* 3, ndf=64,
                    which_model_netD='n_layers',gpu_ids=[0],init_type='normal', n_layers_D=1)

## resume training
idx = 0
if opt.resume:
    print('resume')
    gapath = os.path.join(opt.ckpt_path, opt.resume + '_ga.pth')
    gbpath = os.path.join(opt.ckpt_path, opt.resume + '_gb.pth')
    da1path = os.path.join(opt.ckpt_path, opt.resume + '_da1.pth')
    da2path = os.path.join(opt.ckpt_path, opt.resume + '_da2.pth')
    da3path = os.path.join(opt.ckpt_path, opt.resume + '_da3.pth')
    db1path = os.path.join(opt.ckpt_path, opt.resume + '_db1.pth')
    db2path = os.path.join(opt.ckpt_path, opt.resume + '_db2.pth')
    db3path = os.path.join(opt.ckpt_path, opt.resume + '_db3.pth')

    idx = split = opt.resume.split('_')[1]
    if not any([os.path.isfile(gapath), os.path.isfile(gbpath),
                os.path.isfile(da1path), os.path.isfile(db1path),
                os.path.isfile(da2path), os.path.isfile(db2path),
                os.path.isfile(da3path), os.path.isfile(db3path)]):
        print("=> missing checkpoint files at '{}'".format(opt.resume))

    else:
        print("=> loading checkpoint '{}'".format(opt.resume))

        # g.load_state_dict(torch.load(os.path.join(args.ckpt_path, args.g_snapshot)))
        GA.load_state_dict(torch.load(gapath))
        GB.load_state_dict(torch.load(gbpath))
        DA1.load_state_dict(torch.load(da1path))
        DA2.load_state_dict(torch.load(da2path))
        DA3.load_state_dict(torch.load(da3path))
        DB1.load_state_dict(torch.load(db1path))
        DB2.load_state_dict(torch.load(db2path))
        DB3.load_state_dict(torch.load(db3path))

        opt.ckpt_path = os.path.join(opt.ckpt_path, 'resume_'+ idx)
        if not os.path.exists(opt.ckpt_path):
            os.makedirs(opt.ckpt_path)

GA = GA.cuda()
GB = GB.cuda()
DA1 = DA1.cuda()
DA2 = DA2.cuda()
DA3 = DA3.cuda()
DB1 = DB1.cuda()
DB2 = DB2.cuda()
DB3 = DB3.cuda()

optimizer_G = torch.optim.Adam(itertools.chain(GA.parameters(), GB.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

optimizer_D_A1 = torch.optim.Adam(DA1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_D_B1 = torch.optim.Adam(DB1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

optimizer_D_A2 = torch.optim.Adam(DA2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_D_B2 = torch.optim.Adam(DB2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

optimizer_D_A3 = torch.optim.Adam(DA3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_D_B3 = torch.optim.Adam(DB3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

fake_A_pool = ImagePool(50)
fake_B_pool = ImagePool(50)
fake_A64_pool = ImagePool(50)
fake_B64_pool = ImagePool(50)
fake_A128_pool = ImagePool(50)
fake_B128_pool = ImagePool(50)

Tensor = torch.cuda.FloatTensor if opt.gpu_ids else torch.Tensor

##define losses
criterionGAN = nets.GANLoss(use_lsgan=not opt.no_lsgan, tensor=Tensor)
criterionCycle = torch.nn.L1Loss()
criterionIdt = torch.nn.L1Loss()
criterionRec = torch.nn.L1Loss()
criterionPatch = nets.patchloss()

scale128_transform = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((128,128),interpolation=PIL.Image.BICUBIC),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
scale64_transform = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((64,64),interpolation=PIL.Image.BICUBIC),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])

seg_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])