import os
import math
import itertools
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from tqdm import tqdm

from sparse_linear import sparse_linear

class RuleFunction(torch.autograd.Function):
    '''
    The autograd function used in the Rules Layer.
    The forward function implements the equation (1) in the paper.
    The backward function implements the gradient of the foward function.
    '''
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)

        output = input.mm(weight.t())
        output = output + bias.unsqueeze(0).expand_as(output)
        output = output - (weight * (weight > 0)).sum(-1).unsqueeze(0).expand_as(output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors

        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(input) - grad_output.sum(0).unsqueeze(1).expand_as(weight) * (weight > 0)
        grad_bias = grad_output.sum(0)
        grad_bias[(bias >= 1) * (grad_bias < 0)] = 0

        return grad_input, grad_weight, grad_bias
    
class LabelFunction(torch.autograd.Function):
    '''
    The autograd function used in the OR Layer.
    The forward function implements the equations (4) and (5) in the paper.
    The backward function implements the standard STE estimator.
    '''
    
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)

        output = input.mm((weight.t() > 0).float())
        output += bias.unsqueeze(0).expand_as(output)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors

        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(input)
        grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias
    
class Binarization(torch.autograd.Function):
    '''
    The autograd function for the binarization activation in the Rules Layer.
    The forward function implements the equations (2) in the paper. Note here 0.999999 is used to cancel the rounding error.
    The backward function implements the STE estimator with equation (3) in the paper.
    '''
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = (input > 0.999999).float()
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(input < 0)] = 0
        grad_input[(input >= 1) * (grad_output < 0)] = 0

        return grad_input

class DRNet(nn.Module):
    def __init__(self, in_features, num_rules, out_features):
        """
        DR-Net: https://arxiv.org/pdf/2103.02826.pdf
        
        Args
            in_features (int): the input dimension.
            num_rules (int): number of hidden neurons, which is also the maximum number of rules.
            out_features (int): the output dimension; should always be 1.
        """
        super(DRNet, self).__init__()
        
        self.linear = sparse_linear('l0')
        self.and_layer = self.linear(in_features, num_rules, linear=RuleFunction.apply)
        self.or_layer = self.linear(num_rules, out_features, linear=LabelFunction.apply)

        self.and_layer.bias.requires_grad = False
        self.and_layer.bias.data.fill_(1)
        self.or_layer.weight.requires_grad = False
        self.or_layer.weight.data.fill_(1)
        self.or_layer.bias.requires_grad = False
        self.or_layer.bias.data.fill_(-0.5)
        
    def forward(self, out):
        out = self.and_layer(out)
        out = Binarization.apply(out)
        out = self.or_layer(out)
        
        return out
    
    def regularization(self):
        """
        Implements the Sparsity-Based Regularization (equation 7).
        
        Returns
            regularization (float): the regularization term.
        """
        
        regularization = ((self.and_layer.regularization(axis=1) + 1) * self.or_layer.regularization(mean=False)).mean()
        
        return regularization
    
    def statistics(self):
        """
        Return the statistics of the network.
        
        Returns
            sparsity (float): sparsity of the rule set.
            num_rules (int): number of unpruned rules.
        """
        
        rule_indices = (self.or_layer.masked_weight() != 0).nonzero()[:, 1]
        sparsity = (self.and_layer.masked_weight()[rule_indices] == 0).float().mean().item()
        num_rules = rule_indices.size(0)
        return sparsity, num_rules
            
    def get_rules(self, header=None):
        """
        Translate network into rules.
        
        Args
            header (list OR None): the description of each input feature.
        Returns
            rules (np.array OR list): contains a list of rules. 
                If header is None (2-d np.array), each rule is represented by a list of numbers (1: positive feature, 0: negative feature, 0.5: dont' care).
                If header is not None (list of lists): each rule is represented by a list of strings.
        """
        
        self.eval()
        self.to('cpu')

        prune_weights = self.and_layer.masked_weight()
        valid_indices = self.or_layer.masked_weight().nonzero(as_tuple=True)[1]
        rules = np.sign(prune_weights[valid_indices].detach().numpy()) * 0.5 + 0.5

        if header != None:
            rules_exp = []
            for weight in prune_weights[valid_indices]:
                rule = []
                for w, h in zip(weight, header):
                    if w < 0:
                        rule.append('NOT ' + h)
                    elif w > 0:
                        rule.append(h)
                rules_exp.append(rule)
            rules = rules_exp

        return rules

    def predict(self, X):
        """
        Classifiy the labels of X using rules encoded by the network.
        
        Args
            X (np.array) 2-d np.array of instances with binary features.
        Returns
            results (np.array): 1-d array of labels.
        """
        
        rules = self.get_rules()
        
        results = []
        for x in X:
            indices = np.where(np.absolute(x - rules).max(axis=1) < 1)[0]
            result = int(len(indices) != 0)
            results.append(result)
        return np.array(results)
    
    def save(self, path):
        state = {
            'state_dict': self.state_dict(),
            'parameters': {
                'in_features': self.and_layer.weight.size(1), 
                'num_rules': self.and_layer.bias.size(0), 
                'out_features': self.or_layer.bias.size(0), 
                'and_lam': self.and_lam, 
                'or_lam': self.or_lam,
            }
        }
        
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        torch.save(state, path)
        
    @staticmethod
    def load(path):
        state = torch.load(path)
        model = DRNet(**state['parameters'])
        model.load_state_dict(state['state_dict'])
        
        return model
        
def train(net, train_set, test_set, device="cuda", epochs=2000, batch_size=2000, lr=1e-2, 
          and_lam=1e-2, or_lam=1e-5, num_alter=500):
    def score(out, y):
        y_labels = (out >= 0).float()
        y_corrs = (y_labels == y.reshape(y_labels.size())).float()
        
        return y_corrs
        
    reg_lams = [and_lam, or_lam]
    optimizers = [optim.Adam(net.and_layer.parameters(), lr=lr), optim.Adam(net.or_layer.parameters(), lr=lr)]

    criterion = nn.BCEWithLogitsLoss().to(device)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, drop_last=True, shuffle=True)
    
    with tqdm(total=epochs, desc="Epoch", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}") as t:
        for epoch in range(epochs):
            net.to(device)
            net.train()

            batch_losses = []
            batch_corres = []
            for index, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                out = net(x_batch)
                
                phase = int((epoch / num_alter) % 2)
                loss = criterion(out, y_batch.reshape(out.size())) + reg_lams[phase] * net.regularization()
                optimizers[phase].zero_grad()
                loss.backward()
                optimizers[phase].step()

                corr = score(out, y_batch).sum()

                batch_losses.append(loss.item())
                batch_corres.append(corr.item())
            epoch_loss = torch.Tensor(batch_losses).mean().item()
            epoch_accu = torch.Tensor(batch_corres).sum().item() / len(train_set)

            net.to('cpu')
            net.eval()
            with torch.no_grad():
                test_accu = score(net(test_set[:][0]), test_set[:][1]).mean().item()
                sparsity, num_rules = net.statistics()
                
            t.update(1)
            t.set_postfix({
                'loss': epoch_loss,
                'epoch accu': epoch_accu,
                'test accu': test_accu,
                'num rules': num_rules,
                'sparsity': sparsity,
            })