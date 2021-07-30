import torch

from argparse import ArgumentParser
from time import time

class Options(ArgumentParser):
    """
    Extending the ArgumentParser class to have a couple of utility functions
    """
    def __init__(self):
        super().__init__()
        # basic parameters
        self.add_argument('--root', type=str, default=f'dataset_{round(time())}', help='Directory of images')
        self.add_argument('--border_pad', type=int, default=0, help='Padding between plant and border in multi-plant image in pixels')
        self.add_argument('--l_avg', type=float, default=170.0, help='Value to set l-channel average equal to, larger values are lighter')
        self.add_argument('--a_avg', type=float, default=100.0, help='Value to set a-channel average equal to, smaller values are more green')
        self.add_argument('--b_avg', type=float, default=160.0, help='Value to set b-channel average equal to, larger values are more yellow')
        self.add_argument('--min_scale', type=float, default=0.1, help='Minimum scale of plant relative to background')
        self.add_argument('--max_scale', type=float, default=0.2, help='Maximum scale of plant relative to background')
        self.add_argument('--min_plants', type=int, default=0, help='Minimum number of plants on background')
        self.add_argument('--max_plants', type=int, default=10, help='Maximum number of plants on background')
        self.add_argument('--model_type', type=str, default='unpaired', choices=['paired', 'unpaired'], help='Model to use for image translation')
        self.add_argument('--no_save_all', action='store_true', help='Do not save all intermediate files')
        self.add_argument('--num_images', type=int, default=1, help='Total number of created images')
        self.add_argument('--plant_pad', type=int, default=100, help='Padding between plants in image in pixels')
        self.add_argument('--replace_all', action='store_true', help='Use the entire translated image and not just the bounding box contents')
        # model parameters
        self.add_argument('--gpu_ids', type=str, default='0', help='gpu ids eg 0 0,1 0,1,2, use -1 for CPU')
        self.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        self.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.add_argument('--model', type=str, default='cut', help='chooses which model to use.')
        self.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        self.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        self.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        self.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        self.add_argument('--netG', type=str, default='resnet_9blocks', choices=['resnet_9blocks', 'resnet_6blocks', 'unet_256', 'unet_128', 'stylegan2', 'smallstylegan2', 'resnet_cat'], help='specify generator architecture')
        self.add_argument('--normG', type=str, default='instance', choices=['instance', 'batch', 'none'], help='instance normalization or batch normalization for G')
        self.add_argument('--init_type', type=str, default='xavier', choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='network initialization')
        self.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        self.add_argument('--no_dropout', action='store_false', help='no dropout for the generator')
        self.add_argument('--no_antialias', action='store_true', help='if specified, use stride=2 convs instead of antialiased-downsampling (sad)')
        self.add_argument('--no_antialias_up', action='store_true', help='if specified, use [upconv(learned filter)] instead of [upconv(hard-coded [1,3,3,1] filter), conv]')
        # additional parameters
        self.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        # parameters related to StyleGAN2-based networks
        self.add_argument('--stylegan2_G_num_downsampling',
                            default=1, type=int,
                            help='Number of downsampling layers used by StyleGAN2Generator')
        
    def parse(self):
        """
        Parse the command-line arguments
        """
        args = self.parse_args()
        # check for valid arguments
        if args.border_pad < 0:
            raise ValueError('Border padding cannot be negative')
        if args.min_scale < 0.0:
            raise ValueError('The minimum scale of a single-plant image cannot be negative')
        if args.max_scale > 1.0:
            raise ValueError('The maximum scale of a single-plant image cannot be greater than 1')
        if args.min_scale >= args.max_scale:
            raise ValueError('The minimum scale of a single-plant image cannot be larger than the maximum scale')
        if args.min_plants < 0:
            raise ValueError('The minimum number of plants in an image cannot be negative')
        if args.min_plants > args.max_plants:
            raise ValueError('The minimum number of plants in an image cannot be greater than the maximum')
        if args.plant_pad < 10:
            raise ValueError('Plant padding must be at least 10 pixels, this is needed for lightness matching of the translated image')
        # set gpu ids
        str_ids = args.gpu_ids.split(',')
        args.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                args.gpu_ids.append(id)
        if len(args.gpu_ids) > 0:
            torch.cuda.set_device(args.gpu_ids[0])
        return args
      
    def write_to_file(self, args, path):
        """
        Write the command-line arguments to a text file
        
        Parameters:
            args (argparse.Namespace): object containing all arguments and their values
            path (str): path to save the text file
        """
        with open(path, 'w') as f:
            for arg in args._get_kwargs():
                f.write(f'{arg[0]}: {arg[1]}\n')
