import math
import time

import torch
import torch.nn as nn
import transformers

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class AblateGPT:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

        self.scaler_row = torch.zeros((self.columns), device=self.dev)

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)

        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples

    def get_wanda_mask(self, sparsity, prunen, prunem):
        W_metric = torch.abs(self.layer.weight.data) * torch.sqrt(self.scaler_row.reshape((1,-1)))
        W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
        if prunen != 0:
            for ii in range(W_metric.shape[1]):
                if ii % prunem == 0:
                    tmp = W_metric[:,ii:(ii+prunem)].float()
                    W_mask.scatter_(1,ii+torch.topk(tmp, prunen,dim=1, largest=False)[1], True)
        else:
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity)]
            W_mask.scatter_(1, indices, True)

        return W_mask 

    def get_mag_mask(self, sparsity, prunen, prunem):
        W = self.layer.weight.data 
        W_metric = torch.abs(W)
        if prunen != 0:
            W_mask = (torch.zeros_like(W)==1)
            for ii in range(W_metric.shape[1]):
                if ii % prunem == 0:
                    tmp = W_metric[:,ii:(ii+prunem)].float()
                    W_mask.scatter_(1,ii+torch.topk(tmp, prunen,dim=1, largest=False)[1], True)
        else:
            thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*sparsity)].cpu()
            W_mask = (W_metric<=thresh)

        return W_mask 

    def fasterprune(
        self, args, sparsity, mask=None, prune_n=0, prune_m=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0 or mask is not None: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    # tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    if "wanda" in args.prune_method:
                        tmp = torch.abs(W1) * torch.sqrt(self.scaler_row[i1:i2].reshape((1,-1)))
                    elif "mag" in args.prune_method:
                        tmp = torch.abs(W1)
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0 and mask is None:
                    # tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    if "wanda" in args.prune_method:
                        tmp = torch.abs(W1[:, i:(i+prune_m)]) * torch.sqrt(self.scaler_row[(i+i1):(i+i1+prune_m)].reshape((1,-1)))
                    elif "mag" in args.prune_method:
                        tmp = torch.abs(W1[:, i:(i+prune_m)])
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d 
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()