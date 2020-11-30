import os
import numpy as np
import cv2
import torch
import itertools

def transform(snippet):
    snippet = torch.FloatTensor(snippet).permute(3,0,1,2)
    snippet = snippet.mul(2).sub(255).div(255)
    return snippet

class igazeDataset(torch.utils.data.Dataset):
    def __init__(self, datapath, dataset, mode, data_split, stride=8, len_snippet=24, crop=224):
        self.datapath = datapath
        self.mode = mode
        self.data_split = data_split
        self.stride = stride
        self.len_snippet = len_snippet
        self.crop = crop

        with open(os.path.join(datapath, dataset, '%s_split%d.txt' % (mode, data_split)), 'r') as f:
            self._parse_annts(f.readlines())

        for datapath in ['/scratch/kylemin', '/data/kylemin', '/ss1/kylemin']:
            if os.path.isdir(os.path.join(datapath, dataset)) and os.access(datapath, os.W_OK):
                self.datapath = os.path.join(datapath, dataset)
                break
        print ('datapath for %s: %s' % (self.mode, self.datapath))

    def _parse_annts(self, lines):
        self.cnames, self.labels = [], []
        for l in lines:
            cn, label = l.split(' ')[:2]
            self.cnames.append(cn)
            self.labels.append(int(label)-1)

    def __len__(self):
        return len(self.cnames)

    def __getitem__(self, idx):
        cn, label = self.cnames[idx], self.labels[idx]

        s, e = cn.split('-')[-2:]
        s, e = int(s[1:]), int(e[1:])
        num_frames = e-s
        start_idx = 1
        if self.mode == 'train':
            num_frames = self.len_snippet
            start_idx += np.random.randint(num_frames-self.len_snippet+1)
            pmap = np.zeros((self.len_snippet//self.stride, self.crop, self.crop), np.float32)

            r = np.random.random()
            x1 = np.random.randint(0, 340-self.crop)
            y1 = np.random.randint(0, 256-self.crop)
            for i in range(self.len_snippet//self.stride):
                path_pmap = os.path.join(self.datapath, 'pmaps', cn, '%04d.npy' % (start_idx+i*self.stride))
                if os.path.isfile(path_pmap):
                    p = np.load(path_pmap)
                    p = p[y1:y1+self.crop, x1:x1+self.crop]
                    if np.sum(p) >= 1:
                        if r < 0.5:
                            p = p[:, ::-1]
                        pmap[i] = p

        rgb = []
        flow = []
        for i in range(num_frames):
            rimg = cv2.imread(os.path.join(self.datapath, 'images_rgb', cn, '%04d.jpg' % (start_idx+i)), -1)
            rimg = rimg[..., ::-1]
            fimgu = cv2.imread(os.path.join(self.datapath, 'images_flow', cn, 'u', '%04d.jpg' % (start_idx+i)), -1)
            fimgv = cv2.imread(os.path.join(self.datapath, 'images_flow', cn, 'v', '%04d.jpg' % (start_idx+i)), -1)
            fimg = np.concatenate((fimgu[..., np.newaxis], fimgv[..., np.newaxis]), -1)
            if 'train' in self.mode:
                rimg = rimg[y1:y1+self.crop, x1:x1+self.crop, :]
                fimg = fimg[y1:y1+self.crop, x1:x1+self.crop, :]
                if r < 0.5:
                    rimg = rimg[:, ::-1, :]
                    fimg = fimg[:, ::-1, :]
            rgb.append(rimg)
            flow.append(fimg)

        rgb = transform(rgb)
        flow = transform(flow)

        if self.mode == 'train':
            return rgb, flow, torch.FloatTensor(pmap), label
        else:
            return rgb, flow, label

# Reference: https://github.com/facebookresearch/detectron2
class trainingSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size):
        self.size = size

    def _infinite_indices(self):
        g = torch.Generator()
        while True:
            yield from torch.randperm(self.size, generator=g)

    def __iter__(self):
        yield from itertools.islice(self._infinite_indices(), 0, None, 1)
