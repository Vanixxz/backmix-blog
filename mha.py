import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        assert dim % n_head == 0, "head must be divided by dim"
        self.n_head = n_head
        self.dim = dim
        self.head_dim = n_head // dim
        self.Q = nn.Linear(dim, dim)
        self.K = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)
        self.O = nn.Linear(dim, dim)
        
    def forward(self, x, mask):
        b, l, c = x.shape()
        
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        
        q = q.view(b, l, self.n_head, -1).permute(0,2,1,3).flatten(0,1)
        k = k.view(b, l, self.n_head, -1).permute(0,2,1,3).flatten(0,1)
        v = v.view(b, l, self.n_head, -1).permute(0,2,1,3).flatten(0,1)
        
        attn = torch.bmm(q, k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = torch.bmm(attn, v.transpose(-2,-1))
        attn = attn.view(b, self.n_head, l, -1).permute(0,2,1,3).flatten(-2)
        attn = self.O(attn)
        
        return attn
    
