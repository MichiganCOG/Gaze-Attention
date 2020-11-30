import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_i3d import InceptionI3d, InceptionModule

class I3D_IGA_base(nn.Module):
    def __init__(self):
        super(I3D_IGA_base, self).__init__()
        self.model_rgb = InceptionI3d(modality='rgb')
        self.model_flow = InceptionI3d(modality='flow')

    def forward(self, r, f):
        r4, r5 = self.model_rgb(r)
        f4, f5 = self.model_flow(f)
        x4 = r4+f4
        x5 = r5+f5
        return x4, x5

class I3D_IGA_gaze(nn.Module):
    VALID_ENDPOINTS = ('Mixed_5b', 'Mixed_5c', 'g1', 'b1', 'g2', 'b2', 'g3',)
    def __init__(self):
        super(I3D_IGA_gaze, self).__init__()
        self.add_module('Mixed_5b', InceptionModule(256+320+128+128, [256,160,320,32,128,128], 'Mixed_5b'))
        self.add_module('Mixed_5c', InceptionModule(256+320+128+128, [384,192,384,48,128,128], 'Mixed_5c'))
        self.add_module('g1', nn.Conv3d(1024, 256, kernel_size=(1,3,3), stride=1, padding=(0,1,1)))
        self.add_module('b1', nn.BatchNorm3d(256, eps=0.001, momentum=0.01, affine=True))
        self.add_module('g2', nn.Conv3d(256, 64, kernel_size=(1,3,3), stride=1, padding=(0,1,1)))
        self.add_module('b2', nn.BatchNorm3d(64, eps=0.001, momentum=0.01, affine=True))
        self.add_module('g3', nn.Conv3d(64, 1, kernel_size=1, stride=1, bias=True))

    def forward(self, x):
        for k in self._modules:
            x = self._modules[k](x)
            if k in ['b1', 'b2']:
                x = F.relu(x)
        return x.squeeze(1)

class I3D_IGA_attn(nn.Module):
    def __init__(self, num_action):
        super(I3D_IGA_attn, self).__init__()
        self.fc = nn.Linear(147, 147, bias=True)
        self.dropout = nn.Dropout(0.7)
        self.fc_f = nn.Conv3d(1024, num_action, kernel_size=1, stride=1, bias=True)

    def forward(self, z, h):
        residual = F.avg_pool3d(h, h.shape[2:], stride=1)
        z = self.fc(z.view(z.shape[0],-1)).view(z.shape)
        f = h*torch.sigmoid(z.unsqueeze(1))
        f = F.avg_pool3d(f, f.shape[2:], stride=1)
        f += residual
        f = self.dropout(f)
        f = self.fc_f(f)
        return f.view(f.shape[:2])
