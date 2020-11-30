import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def get_accuracy(cm):
    list_acc = []
    for i in range(len(cm)):
        acc = 0
        if cm[i,:].sum() > 0:
            acc = cm[i,i]/cm[i,:].sum()
        list_acc.append(acc)

    return 100*np.mean(list_acc), 100*np.trace(cm)/np.sum(cm)

def sample_gumbel(shape, device, eps):
    U = torch.rand(shape, device=device)
    return -torch.log(eps - torch.log(U + eps))

def make_hard_decision(pi, device, eps=1e-10):
    pir = pi.view(pi.shape[0]*pi.shape[1], pi.shape[2]*pi.shape[3])
    gumbel_noise = sample_gumbel(pir.shape, device, eps)
    pi_g = pir + gumbel_noise
    k = pi_g.max(-1)[1]
    z_hard = torch.zeros(pir.shape, device=device).scatter_(-1, k.view(-1,1), 1)

    return z_hard.view(pi.shape), pi_g.view(pi.shape)

def compute_cross_entropy(z, h, model_attn, label):
    y = model_attn(z, h)
    return y, F.cross_entropy(y, label, reduction='none')

# Reference: https://github.com/GuyLor/Direct-VAE
def compute_gradients_gaze(z_hard, h, model_attn, pi_g, label, device, eps, z_pre_realized=None):
    with torch.no_grad():
        N, height, width = z_hard.shape[1:]
        soft_copy = pi_g.view(-1, height*width)
        hard_copy = z_hard
        model_attn.eval()

        if z_pre_realized is None:
            tlist_y = torch.LongTensor(range(height))
            tlist_x = torch.LongTensor(range(width))
            list_idx = torch.stack(torch.meshgrid(tlist_y, tlist_x), -1).view(-1, 2)
            z_pre_realized = torch.zeros((1, list_idx.shape[0], height, width), device=device)
            for i, (j, k) in enumerate(list_idx):
                z_pre_realized[0][i][j][k] = 1
            z_pre_realized = z_pre_realized.repeat(hard_copy.shape[0],1,1,1).view(-1,height,width)

        hard_tmp = hard_copy.unsqueeze(1).repeat(1,height*width,1,1,1)
        idx_b = torch.LongTensor(range(hard_copy.shape[0])).unsqueeze(1).repeat(1,height*width).view(-1)
        idx_K = torch.LongTensor(range(height*width)).repeat(hard_copy.shape[0])
        list_idx_N = torch.LongTensor(range(N)).repeat(hard_copy.shape[0]*height*width,1).t()
        batch_z_new = []
        for n in range(N):
            hard_tmp_new = hard_tmp.clone()
            hard_tmp_new[idx_b,idx_K,list_idx_N[n]] = z_pre_realized
            batch_z_new.append(hard_tmp_new)

        batch_z_new = torch.cat(batch_z_new, 1).view(-1, N, height, width)
        label_new = label.unsqueeze(-1).repeat(1,N*height*width).view(-1)

        div = N
        sb = batch_z_new.shape[0]//div
        hr = N*height*width//div
        list_losses = []
        for d in range(div):
            losses = compute_cross_entropy(batch_z_new[d*sb:d*sb+sb], h.repeat(1,hr,1,1,1).view(-1,h.shape[1],N,height,width), model_attn, label_new[d*sb:d*sb+sb])[1]
            list_losses.append(losses)
        losses = torch.cat(list_losses, 0)
        losses = eps*losses.view(-1, height*width)
        soft_copy = soft_copy - losses
        k = soft_copy.max(-1)[1]

        change = torch.zeros(soft_copy.shape, device=device)
        change.scatter_(-1, k.view(-1, 1), 1)
        gradients = hard_copy.view(-1, height*width) - change
        model_attn.train()

        return gradients.view(-1,N,height,width)/eps

def _pointwise_loss(lambd, inp, trg, reduction):
    d = lambd(inp, trg)
    if reduction == 'none':
        return d
    return torch.mean(d) if reduction=='mean' else torch.sum(d)
