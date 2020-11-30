import os
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from dataset import igazeDataset, trainingSampler
from model import I3D_IGA_base, I3D_IGA_gaze, I3D_IGA_attn
from utils import get_accuracy, make_hard_decision, compute_cross_entropy, compute_gradients_gaze
import time
from datetime import timedelta
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description='igaze')
parser.add_argument('--mode', default='test', help='train | test')
parser.add_argument('--crop', type=int, default=224, help='for spatial cropping')
parser.add_argument('--trange', type=int, default=24, help='temporal range')
parser.add_argument('--stride', type=int, default=8, help='pooling stride for gaze prediction')
parser.add_argument('--b', type=int, default=1, help='batch size')
parser.add_argument('--wd', type=float, default=4e-5, help='weight decay')
parser.add_argument('--it1', type=int, default=8000, help='first decay point')
parser.add_argument('--it2', type=int, default=15000, help='second decay point')
parser.add_argument('--iters', type=int, default=18000, help='number of max iterations for training')
parser.add_argument('--lr', type=float, default=0.032, help='learning rate')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--eps', type=float, default=1000, help='epsilon for the gradient estimator')
parser.add_argument('--anneal', type=float, default=1e-3, help='anneal rate for epsilon')

parser.add_argument('--datapath', default='dataset', help='path to dataset')
parser.add_argument('--datasplit', type=int, default=1, help='data split for the cross validation')
parser.add_argument('--weight', default='weights/i3d_iga_best1_base.pt', help='path to the weight file for the base network')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--sparse', type=int, default=0, help='whether to test sparsely for fast evaluation: True (1) | False (0)')


def main():
    global args, device
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    exp_name = '%d_%d_%d_%s_%d' % (args.crop, args.trange, args.stride, time.strftime("%m-%d_%H-%M-%S"), args.seed)
    if args.mode == 'test':
        if os.path.isfile(args.weight):
            exp_name = 'test_'+'.'.join(args.weight.split('/')[-1].split('.')[:-1])
        else:
            raise ValueError('unknown weight: '+args.weight)

    num_action = 106
    if args.mode == 'train':
        dataset = igazeDataset(args.datapath, 'EGTEA', args.mode, args.datasplit, args.stride, args.trange, args.crop)
        train_loader = DataLoader(dataset, num_workers=4*args.ngpu, batch_size=args.b, sampler=trainingSampler(len(dataset)))
    else:
        test_loader = DataLoader(igazeDataset(args.datapath, 'EGTEA', args.mode, args.datasplit, args.stride), num_workers=4, pin_memory=True)

    print_args(exp_name)
    model_base, model_gaze, model_attn = load_model(num_action)
    optimizer = load_weights_and_set_opt(model_base, model_gaze, model_attn)

    if torch.cuda.is_available():
        print ('run on cuda')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda")
        if args.ngpu > 1:
            model_base = torch.nn.DataParallel(model_base, device_ids=range(args.ngpu))
            model_gaze = torch.nn.DataParallel(model_gaze, device_ids=range(args.ngpu))
            model_attn = torch.nn.DataParallel(model_attn, device_ids=range(args.ngpu))
    else:
        print ('run on cpu')
        device = torch.device("cpu")

    model_base = model_base.to(device)
    model_gaze = model_gaze.to(device)
    model_attn = model_attn.to(device)

    if args.mode == 'train':
        train(train_loader, model_base, model_gaze, model_attn, optimizer, exp_name)
    else:
        test(test_loader, model_base, model_gaze, model_attn, num_action)


def load_model(num_action):
    model_base = I3D_IGA_base()
    model_gaze = I3D_IGA_gaze()
    model_attn = I3D_IGA_attn(num_action)

    return model_base, model_gaze, model_attn


def load_weights(model, weight_file):
    if os.path.isfile(weight_file):
        print ('loading weight file: %s' % weight_file)
        weight_dict = torch.load(weight_file)
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    print (' size? ' + name, param.size(), model_dict[name].size())
            else:
                print (' name? ' + name)
    else:
        print ('no weight file: %s ... start from scratch' % weight_file)


def load_weights_and_set_opt(model_base, model_gaze, model_attn):
    load_weights(model_base, args.weight)
    load_weights(model_gaze, args.weight.replace('base', 'gaze'))
    load_weights(model_attn, args.weight.replace('base', 'attn'))

    params = []
    for model in [model_base, model_gaze, model_attn]:
        params.append({'params': model.parameters()})
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    return optimizer


def print_args(exp_name):
    print ('exp_name:      %s' % exp_name)
    print ('datasplit:     %d' % args.datasplit)
    print ('weight:        %s' % args.weight)
    print ('mode:          %s' % args.mode)
    if args.mode == 'train':
        print ('ngpu:          %d' % args.ngpu)
        print ('b:             %d' % args.b)
        print ('iters:         %d, %d, %d' % (args.it1, args.it2, args.iters))
        print ('lr:            %g' % args.lr)
        print ('wd:            %g' % args.wd)
        print ('eps, anneal:   %g, %g' % (args.eps, args.anneal))
    else:
        print ('sparse:        %d' % args.sparse)


def adjust_lr(optimizer, step):
    if step in [args.it1, args.it2]:
        for opt in optimizer.param_groups:
            opt['lr'] *= 0.1


def train(train_loader, model_base, model_gaze, model_attn, optimizer, exp_name):
    path_output = os.path.join('output', exp_name)
    if not os.path.isdir(path_output):
        os.makedirs(path_output)

    eps = args.eps
    scale = 32
    slen = args.crop//scale
    tlist_y = torch.LongTensor(range(slen))
    tlist_x = torch.LongTensor(range(slen))
    list_idx = torch.stack(torch.meshgrid(tlist_y, tlist_x), -1).view(-1, 2)
    z_pre_realized = torch.zeros((1, list_idx.shape[0], slen, slen), dtype=torch.float32, device=device)
    for i, (j, k) in enumerate(list_idx):
        z_pre_realized[0][i][j][k] = 1
    z_pre_realized = z_pre_realized.repeat(args.b,1,1,1).view(-1,slen,slen)

    start_time = time.time()
    for i, (rgb, flow, pmap, label) in zip(range(1, args.iters+1), train_loader):
        rgb, flow, pmap, label = rgb.to(device), flow.to(device), pmap.to(device), label.to(device)
        pi, h = model_base(rgb, flow)
        pi = model_gaze(pi)

        k = pmap.shape[-1]//pi.shape[-1]
        pmap = F.max_pool2d(pmap, kernel_size=(k,k), stride=(k,k))

        loss_kl = torch.tensor(0., requires_grad=True)
        idx_valid = torch.sum(pmap, dim=[2,3])>0
        num_valid = idx_valid.sum()
        if num_valid > 0:
            pmap = pmap[idx_valid].view(num_valid, -1)
            pi_valid = pi[idx_valid].view(num_valid, -1)
            loss_kl = F.kl_div(F.log_softmax(pi_valid, dim=1), pmap/torch.sum(pmap, dim=1, keepdim=True), reduction='batchmean')

        z_hard, pi_g = make_hard_decision(pi, device)
        y, loss_cn = compute_cross_entropy(z_hard, h, model_attn, label)
        loss_cn = loss_cn.mean()
        gradients = compute_gradients_gaze(z_hard, h, model_attn, pi_g, label, device, eps, z_pre_realized)
        loss_attn = (gradients*pi_g).mean(0).sum()

        loss = loss_cn + loss_kl + loss_attn

        optimizer.zero_grad()
        loss.backward()

        params = []
        params += list(model_base.parameters())
        params += list(model_gaze.parameters())
        params += list(model_attn.parameters())
        grad_total = clip_grad_norm_(params, 20)
        optimizer.step()

        adjust_lr(optimizer, i)
        if i % 100 == 0:
            eps = max(0.1, args.eps*np.exp(-args.anneal*i))

        print ('step: [%5d/%5d], %s' % (i, args.iters, timedelta(seconds=int(time.time()-start_time))), flush=True)

        if i % 500 == 0 and i >= 10000: # in this implementation, the model performs best after about 10000~15500 iterations
            torch.save(model_base.state_dict(), os.path.join(path_output, '%s_%05d_base.pt' % (exp_name, i)))
            torch.save(model_gaze.state_dict(), os.path.join(path_output, '%s_%05d_gaze.pt' % (exp_name, i)))
            torch.save(model_attn.state_dict(), os.path.join(path_output, '%s_%05d_attn.pt' % (exp_name, i)))


def test(test_loader, model_base, model_gaze, model_attn, num_action):
    model_base.eval()
    model_gaze.eval()
    model_attn.eval()

    list_true = []
    list_pred = []

    start_time = time.time()
    with torch.no_grad():
        for i, (rgb, flow, label) in enumerate(test_loader, 1):
            label = label.to(device)
            len_video, height, width = rgb.shape[2:]
            top, left = (height-args.crop)//2, (width-args.crop)//2
            jump = args.trange
            if args.sparse:
                if len_video > args.trange*10:
                    jump = len_video // 10

            list_start_idx = list(range(0, len_video-args.trange+1, jump))

            list_y = []
            for t in list_start_idx:
                t_rgb = rgb[..., t:t+args.trange, top:top+args.crop, left:left+args.crop].cuda()
                t_flow = flow[..., t:t+args.trange, top:top+args.crop, left:left+args.crop].cuda()
                pi, h = model_base(t_rgb, t_flow)
                pi = model_gaze(pi)
                z_hard, pi_g = make_hard_decision(pi, device)
                y = compute_cross_entropy(z_hard, h, model_attn, label)[0]
                list_y.append(y)

            y_avg = torch.cat(list_y, 0).mean(0, keepdim=True)

            list_true.append(label.item())
            list_pred.append(torch.argmax(y_avg, 1).item())

            print ('step: %04d, %s' % (i, timedelta(seconds=int(time.time()-start_time))), flush=True)

    mean_class_acc, acc = get_accuracy(confusion_matrix(list_true, list_pred, labels=list(range(num_action))))

    print ('acc: %.2f, %.2f / %s' % (mean_class_acc, acc, timedelta(seconds=int(time.time()-start_time))), flush=True)


if __name__ == '__main__':
    main()
