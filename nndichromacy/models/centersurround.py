import torch
from torch import nn

def make_mask_grid(h, w, outdims):
    if h > w:
        yy, xx = torch.meshgrid(
            [
                torch.linspace(-max(h, w) / min(h, w), max(h, w) / min(h, w), h),
                torch.linspace(-1, 1, w),
            ]
        )
    else:
        yy, xx = torch.meshgrid(
            [
                torch.linspace(-1, 1, h),
                torch.linspace(-max(h, w) / min(h, w), max(h, w) / min(h, w), w),
            ]
        )
    grid = torch.stack([xx, yy], 2)[None, ...]
    return grid.repeat([outdims, 1, 1, 1])

def sigmoid(z, temp=1):
    return 1/(1+torch.exp(-z/temp))

def normalize(dd, lower=0, upper=1):
    dd_ms = dd - dd.min()
    return dd_ms / dd_ms.max() * (upper - lower) + lower

class Center(nn.Module):
    def __init__(self, h, w, outdims, init_weight_center=1.0, init_width=.3, center_weights_upper=None, center_weights_lower=None, mask_weight_fix=False, temp=0.1):
        super().__init__()
        self.h = h
        self.w = w
        self.outdims = outdims
        self.init_width = init_width
        self.temp = temp

        self._mu = nn.Parameter(torch.zeros(outdims, 2))
        self._width = nn.Parameter(torch.ones(outdims) * init_width)
        if not mask_weight_fix:
            self._weights = nn.Parameter(torch.zeros(outdims) + init_weight_center + 1e-3)
        else: self._weights = (torch.zeros(outdims) + init_weight_center + 1e-3).cuda()

        self.center_weights_upper = center_weights_upper
        self.center_weights_lower = center_weights_lower

    @property
    def mu(self):
        self._mu.data.clamp_(-1, 1)
        return self._mu
    
    @property
    def width(self):
        self._width.data.clamp_(1e-3)
        return self._width

    @property   
    def weights(self):
        if self.center_weights_lower != None or self.center_weights_upper != None:
            self._weights.data.clamp_(self.center_weights_lower, self.center_weights_upper)
        return self._weights
    
    @staticmethod
    def generate_disk(h, w, outdims, mean, std, temp=0.01):
        grid = make_mask_grid(h, w, outdims).cuda()
        mean = mean.reshape(outdims, 1, 1, -1)
        std = std.reshape(outdims, 1, 1, 1)

        pdf = grid - mean
        pdf = torch.sum((pdf/std) ** 2, dim=-1)
        pdf = torch.exp(-0.5 * pdf)
        pdf = normalize(pdf, lower=-1, upper=1)
        disk = sigmoid(pdf, temp=temp)
        return disk
    
    def forward(self, shift=None):
        mu = self.mu + shift[None, ...] if shift is not None else self.mu
        masks = self.generate_disk(self.h, self.w, self.outdims, mu, self.width, temp=self.temp)
        
        # Make sure this is a probability distribution
        masks_pdf = masks / torch.sum(masks, dim=(1, 2), keepdim=True)
        return masks_pdf * self.weights.view(-1, 1, 1)
    
    
class Surround(nn.Module):
    def __init__(self, h, w, outdims, init_weight_surround=-1.0, init_width_inner=.2, init_width_outer=.4, dog=True, 
                 surround_weights_upper=None, surround_weights_lower=None, mask_weight_fix=False, cs_share_loc=False, center_mu=None, temp=0.1):
        super().__init__()
        self.h = h
        self.w = w
        self.outdims = outdims
        self.init_width_inner = init_width_inner
        self.init_width_outer = init_width_outer
        self.temp = temp
        self.dog = dog # boolean; if False, surround is a gaussian

        if init_width_outer < init_width_inner:
            raise ValueError("Width of outer Gaussian disk cannot be smaller than the inner disk.")
        
        self._mu = nn.Parameter(torch.zeros(outdims, 2))
        if not mask_weight_fix:
            self._weights = nn.Parameter(torch.zeros(outdims) + init_weight_surround + 1e-3) # initialize at -1 gives best results in 4 layer model
        else: 
            self._weights = (torch.zeros(outdims) + init_weight_surround + 1e-3).cuda() 

        self._width_outer = nn.Parameter(torch.ones(outdims) * init_width_outer)
        
        if dog:
            self._width_inner = nn.Parameter(torch.ones(outdims) * init_width_inner)
            self._outer_weights = nn.Parameter(torch.zeros(outdims) - 5.)
        
        self.surround_weights_upper = surround_weights_upper
        self.surround_weights_lower = surround_weights_lower

    @property
    def mu(self):
        self._mu.data.clamp_(-1, 1)
        return self._mu
    
    @property
    def width_inner(self):
        self._width_inner.data.clamp_(1e-3,1.5)
        return self._width_inner
    
    @property
    def width_outer(self):
        if self.dog:
            width_outer = (self._width_outer - self._width_inner).clamp(0.) + self._width_inner
            return width_outer
        else:
            self._width_outer.data.clamp_(1e-3)
            return self._width_outer
        ## the code below will perfomr the above operation with gradients


    @property
    def outer_weights(self):
        outer_weights = torch.sigmoid(self._outer_weights) / 2. + .5
        return outer_weights
    
    @property
    def inner_weights(self):
        inner_weights = 1 - self.outer_weights
        return inner_weights
    
    @property
    def weights(self):
        if self.surround_weights_lower != None or self.surround_weights_upper != None:
            self._weights.data.clamp_(self.surround_weights_lower, self.surround_weights_upper)
        return self._weights
    
    @staticmethod
    def generate_disk(h, w, outdims, mean, std, temp=0.01):
        grid = make_mask_grid(h, w, outdims).cuda()
        mean = mean.reshape(outdims, 1, 1, -1)
        std = std.reshape(outdims, 1, 1, 1)

        pdf = grid - mean
        pdf = torch.sum((pdf/std) ** 2, dim=-1)
        pdf = torch.exp(-0.5 * pdf)
        pdf = normalize(pdf, lower=-1, upper=1)
        disk = sigmoid(pdf, temp=temp)
        
        return disk
    
    def forward(self, shift=None, center_pos=None):
        #print('center_pos',center_pos)
        mu = center_pos if center_pos is not None else self.mu
        mu = mu + shift[None, ...] if shift is not None else mu
        #print('surround',mu)
        outer_masks = self.generate_disk(self.h, self.w, self.outdims, mu, self.width_outer, temp=self.temp)
        
        if self.dog:
            inner_masks = self.generate_disk(self.h, self.w, self.outdims, mu, self.width_inner, temp=self.temp)
            masks = outer_masks * self.outer_weights.view(-1, 1, 1) - inner_masks * self.inner_weights.view(-1, 1, 1)
        
        else:
            masks = outer_masks
        
        # Make sure this is a probability distribution
        masks_pdf = masks / torch.sum(masks, dim=(1, 2), keepdim=True)
        
        return masks_pdf * self.weights.view(-1, 1, 1)
