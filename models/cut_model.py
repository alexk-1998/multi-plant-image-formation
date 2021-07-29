import os
import torch

from . import networks


class CUTModel():

    def __init__(self, opt):
        """Initialize the CUTModel class."""
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        torch.backends.cudnn.benchmark = True
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.load_networks()
        self.netG.eval()

    def load_networks(self):
        """Load the generator network."""
        p = os.path.join('models', 'models', self.opt.model_type.lower()+'_G.pth')
        state_dict = torch.load(p, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        self.netG.load_state_dict(state_dict)
