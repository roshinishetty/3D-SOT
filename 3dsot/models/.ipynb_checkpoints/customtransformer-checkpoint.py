## Our transformers codebase

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
import math
from pointnet2.utils import pytorch_utils as pt_utils
import torch.nn.functional as F

class AttentionHead(nn.Module):
    def __init__(self, d_model=512, d_internal=64, dropout=0.1):
        """
        :param d_model: input and output dim
        :param d_internal: query and key dimension
        :param dropout
        """
        super().__init__()
        self.d_model = d_model
        self.d_internal = d_internal
        self.query = nn.Linear(d_model, d_internal)
        self.key = nn.Linear(d_model, d_internal)
        self.value = nn.Linear(d_model, d_model)
        
        self.softmax = nn.Softmax(dim=-1)
#         self.linear = nn.Linear(d_model, d_model)
#         self.relu = nn.ReLU()
        
#         self.linear2 = nn.Linear(d_model, d_internal)
#         self.relu2 = nn.ReLU()
#         self.dropout2 = nn.Dropout(dropout)
#         self.linear3 = nn.Linear(d_internal, d_model)
#         self.dropout3 = nn.Dropout(dropout)
#         self.layernorm3 = nn.LayerNorm(d_model)
        

    def forward(self, query, key, value):
        """
        :param query
        :param key
        :param value
        :return: a tuple of two elements:
            - attention scores aka final outputs
            - attention probabilities matrix
        """
        #n_pixels, batch, dim
        q = self.query(query).permute(1, 0, 2) # batch, n_pixels, dim 
        k = self.key(key).permute(1, 0, 2) # batch, m_pixels, dim 
        v = self.value(value).permute(1, 0, 2) # batch, m_pixels, dim 
        q_k = torch.matmul(q, k.transpose(1,2)) # batch, n_pixels, m_pixels
        q_k /= self.d_internal**0.5
        probs = self.softmax(q_k)
        probs = probs/(1e-9 + probs.sum(dim=1, keepdim=True))
        aten_scores = torch.matmul(probs, v).permute(1,0,2)
#         res_con = aten_scores + query
#         aten_weights = self.linear(res_con)

#         aten_weights = self.relu(aten_weights)
#         aten_weights2 = self.linear2(aten_weights)
#         aten_weights2 = self.relu2(aten_weights2)
#         aten_weights2 = self.dropout2(aten_weights2)
#         aten_weights2 = self.linear3(aten_weights2)
#         aten_weights2 = self.dropout3(aten_weights2)
#         aten_weights = aten_weights2 + aten_weights
#         aten_weights = self.layernorm3(aten_weights)

        return aten_scores, probs

class FeedForward(nn.Module):
    """
        :param d_model: input and output dim
        :param d_internal: query and key dimension
        :param dropout
    """
    def __init__(self, d_model=512, d_internal=64, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(d_model, d_internal)
        self.linear2 = nn.Linear(d_internal, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.instancenorm = nn.InstanceNorm1d(d_model)
        self.relu2 = nn.ReLU()
        
    def forward(self, input):
        out = self.linear(input)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout2(out)
        out = out + input
        
        out = self.instancenorm(out.permute(1, 2, 0))
        out = self.relu2(out.permute(2, 0, 1))
        return out
        


class MultiheadAttention(nn.Module):
    def __init__(self, num_layers=1, d_model=512, d_internal=64):
        """
        :param num_layers: number of attention heads
        :param d_model: input and output dim
        :param d_internal: query and key dimension
        """
        super().__init__()
        self.num_layers = num_layers
        self.attention_heads = nn.ModuleList()
        for i in range(num_layers):
            self.attention_heads.append(AttentionHead(d_model, d_internal))
                     
        self.linear = nn.Linear(d_model, d_model)
        self.relu = nn.ReLU()
        self.layernorm = nn.LayerNorm(d_model)
        
        self.feedforward = FeedForward(d_model, d_internal)

    def forward(self, q, k, v):
        """

        :param q
        :param k
        :param v
        :return: attention output and probs
        """
        # inp = self.emb(indices)
        aten_scores = None
        first_layer = True
        for attention_head in self.attention_heads:
            if first_layer:
                aten_scores, probs = attention_head(q, k, v)
                concat = aten_scores
                first_layer = False
            else:
                aten_scores = attention_head(q,k,v)
                concat = torch.cat((concat, aten_scores), -1)

        out = self.linear(concat)
        out = self.relu(out)
        out = out + q
        out = self.layernorm(out)
        
        out = self.feedforward(out)

        return out, probs
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int=256, num_positions: int=3, batched=False):
        """
        :param d_model: input and output dim of Attention head
        :param num_positions: input dim
        """
        super().__init__()
        self.conv = nn.Conv1d(num_positions, d_model, kernel_size=1)
        self.batchnorm = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x):
        """
        :param x: input for pos embeddings computation
        """
        # B, n_pixels, 3
        x = x.transpose(1,2).contiguous()
        out = self.conv(x)
        out = self.batchnorm(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out
    
class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, d_internal=128, n_heads=1, n_layers=1):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(n_heads=n_heads, d_model=d_model,d_internal=d_internal)
        self.encoder_layers = nn.ModuleList([encoder_layer for i in range(n_layers)])
        
    def forward(self, inp):
        intermediate = inp
        for encoder in self.encoder_layers:
            intermediate = encoder(intermediate)
        return intermediate
        
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_internal=128, n_heads=1):
        super().__init__()
        self.self_attention = MultiheadAttention(num_layers=n_heads, d_internal=d_internal,d_model=d_model)
        
        
    def forward(self,inp):
        return self.self_attention(inp, inp, inp)[0]
        
        
class TransformerDecoder(nn.Module):
    def __init__(self, d_model=256, d_internal=128, n_heads=1, n_layers=1):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(n_heads=n_heads, d_model=d_model,d_internal=d_internal)
        self.decoder_layers = nn.ModuleList([decoder_layer for i in range(n_layers)])
        
    def forward(self, inp_decoder, kv_encoder):
        intermediate = inp_decoder
        for decoder in self.decoder_layers:
            intermediate, probs = decoder(intermediate, kv_encoder)
        return intermediate, probs
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_internal=128, n_heads=1):
        super().__init__()
        self.self_attention = MultiheadAttention(num_layers=n_heads, d_internal=d_internal,d_model=d_model)
        self.cross_attention = MultiheadAttention(num_layers=n_heads, d_internal=d_internal,d_model=d_model)
        
        
    def forward(self, inp, kv_encoder):
        out = self.self_attention(inp, inp, inp)[0]
        res, probs = self.cross_attention(out, kv_encoder, kv_encoder)
        return res, probs
        
class TransformerFusion(nn.Module):
    def __init__(self, num_layers_encoder = 1, num_layers_decoder=1, d_model=256, d_internal=128, n_heads=1):
        super().__init__()
        self.transformer_encoder = TransformerEncoder(d_model=d_model, d_internal=d_internal, n_heads=n_heads, n_layers=num_layers_encoder)
        self.transformer_decoder = TransformerDecoder(d_model=d_model, d_internal=d_internal, n_heads=n_heads, n_layers=num_layers_decoder)
        self.pos_emb_enc = PositionalEncoding(d_model=d_model, num_positions=3)
        self.pos_emb_dec = PositionalEncoding(d_model=d_model, num_positions=3)
        self.feature_convs = (pt_utils.Seq(256)
                .conv1d(256, bn=True)
                .conv1d(256, activation=None))
        
    def forward(self, search_feature, search_xyz, template_feature, template_xyz):
        search_feature = search_feature.permute(2,0,1) + self.pos_emb_dec(search_xyz).permute(2, 0, 1)
        template_feature = template_feature.permute(2,0,1) + self.pos_emb_enc(template_xyz).permute(2,0,1)
        kv_encoder = self.transformer_encoder(template_feature)
        out, probs = self.transformer_decoder(search_feature, kv_encoder)
        out = out.permute(1,2,0)
        out = self.feature_convs(out)
        return out, probs