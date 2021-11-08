import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim

def sparse_linear(name):
    if name == 'linear':
        return Linear
    elif name == 'l0':
        return L0Linear
    elif name == 'reweight':
        return ReweightLinear
    else:
        raise ValueError(f'{name} linear type not supported.')

class Linear(nn.Linear):
    def __init__(self, in_features, out_features,  bias=True, linear=F.linear, **kwargs):
        super(Linear, self).__init__(in_features, out_features, bias=bias, **kwargs)
        
        self.linear = linear
        
    def forward(self, input):
        output = self.linear(input, self.weight, self.bias)
        
        return output
    
    def sparsity(self):
        sparsity = (self.weight == 0).float().mean().item()
        
        return sparsity
    
    def masked_weight(self):
        masked_weight = self.weight
        
        return masked_weight
    
    def regularization(self):
        regularization = 0
        
        return regularization

class L0Linear(nn.Linear):
    def __init__(self, in_features, out_features,  bias=True, linear=F.linear, loc_mean=0, loc_sdev=0.01, 
                 beta=2 / 3, gamma=-0.1, zeta=1.1, fix_temp=True, **kwargs):
        super(L0Linear, self).__init__(in_features, out_features, bias=bias, **kwargs)
        
        self._size = self.weight.size()
        self.loc = nn.Parameter(torch.zeros(self._size).normal_(loc_mean, loc_sdev))
        self.temp = beta if fix_temp else nn.Parameter(torch.zeros(1).fill_(beta))
        self.register_buffer("uniform", torch.zeros(self._size))
        self.gamma = gamma
        self.zeta = zeta
        self.gamma_zeta_ratio = math.log(-gamma / zeta)
        self.linear = linear
        
        self.penalty = 0

    def forward(self, input):
        mask, self.penalty = self._get_mask()
        masked_weight = self.weight * mask
        output = self.linear(input, masked_weight, self.bias)
        
        return output
    
    def sparsity(self):
        sparsity = (self.masked_weight() == 0).float().mean().item()
        
        return sparsity
    
    def masked_weight(self):
        mask, _ = self._get_mask()
        masked_weight = self.weight * mask
        
        return masked_weight
    
    def regularization(self, mean=True, axis=None):
        regularization = self.penalty
        if mean:
            regularization = regularization.mean() if axis == None else regularization.mean(axis)

        return regularization
    
    def _get_mask(self):
        def hard_sigmoid(x):
            return torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x))

        if self.training:
            self.uniform.uniform_()
            u = torch.autograd.Variable(self.uniform)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.loc) / self.temp)
            s = s * (self.zeta - self.gamma) + self.gamma
            penalty = torch.sigmoid(self.loc - self.temp * self.gamma_zeta_ratio)
        else:
            s = torch.sigmoid(self.loc) * (self.zeta - self.gamma) + self.gamma
            penalty = 0
            
        return hard_sigmoid(s), penalty
    
class ReweightLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, linear=F.linear, 
                 prune_neuron=False, prune_always=True, factor=0.1):
        super(ReweightLinear, self).__init__(in_features, out_features, bias=bias)
        
        self.prune_neuron = prune_neuron
        self.prune_always = prune_always
        self.factor = factor
        self.linear = linear

    def forward(self, input):
        if self.eval():
            weight = self.masked_weight()
        else:
            weight = self.masked_weight() if self.prune_always else self.weight
        out = self.linear(input, weight, self.bias)
        
        return out
    
    def sparsity(self):
        sparsity = (self.weight.abs() <= self._threshold()).float().mean().item()
        
        return sparsity
    
    def masked_weight(self):
        masked_weight = self.weight.clone()
        masked_weight[self.weight.abs() <= self._threshold()] = 0
        
        return masked_weight

    def regularization(self, mean=True, axis=None):
        regularization = self.weight.abs()
        if mean:
            regularization = regularization.mean() if axis == None else regularization.mean(axis)
            
        return regularization
    
    def _threshold(self):
        if self.prune_neuron:
            threshold = self.factor * self.weight.std(1).unsqueeze(1)
        else:
            threshold = self.factor * self.weight.std()
        
        return threshold