from sklearn.metrics import log_loss
import torch.nn as nn
import torch
import math
import numpy as np  
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from .transformer import *
import torchvision.models as models
from einops import rearrange, repeat
from transformers import AutoModel
import random

# import pytorch_lightning as pl
from contextlib import contextmanager
from copy import deepcopy
import json
from transformers import AutoTokenizer
from transformers import AutoModel
import torchvision.models as models
from einops import rearrange, repeat
import random
from .model_SepMedKLIP import *
# from scheduler import create_scheduler
# from optim import create_optimizer



class SepMedKLIP_V5(SepMedKLIP):

    def __init__(self, config, ana_book, disease_book, mode='train'):
        super().__init__(config, ana_book, disease_book, mode)

        self.loc_fuse_module = LlamaMLP(self.d_model*2, self.d_model*2, self.d_model)


    def get_train_loss(self,
                       loc_status_pred, loc_labels,
                       loc_common_feat, loc_sim,
                       dis_status_pred, dis_labels,
                       feat_orthog_score,
                       full_labels,
                       radgraph_full_labels=None):
        
        batch_size = loc_labels.shape[0]
        loc_num = self.ana_book.shape[0]
        dis_num = self.disease_book.shape[0]

        loss_loc_status =  F.cross_entropy(loc_status_pred.reshape(-1,loc_status_pred.shape[-1]), 
                                           loc_labels.reshape(-1,1)[:,0], reduction='none')
        loss_loc_status = rearrange(loss_loc_status, '(b l) -> b l', b=batch_size)
        loss_loc_status = torch.mean(loss_loc_status)

        loc_sim_gt = torch.arange(loc_num).type_as(loc_sim).long()  # [loc_num];
        loc_sim_gt = repeat(loc_sim_gt, 'i -> b i', b=batch_size)   # [bz, loc_num];
        loss_loc_common = F.cross_entropy(loc_sim.transpose(1,2), loc_sim_gt, reduction='none').mean()


        dis_status_pred = dis_status_pred.reshape(-1, dis_status_pred.shape[-1])
        dis_labels_ = dis_labels.reshape(-1, 1)
        loss_dis_status = F.cross_entropy(dis_status_pred, dis_labels_[:,0], reduction='none')
        loss_dis_status = rearrange(loss_dis_status, '(b d) -> b d', b=batch_size)
        loss_dis_status = torch.mean(loss_dis_status)

        inverted_mask = (radgraph_full_labels != 1)
        masked_feat_orthog_score = feat_orthog_score[inverted_mask]
        loss_orthog = torch.mean(torch.clamp(masked_feat_orthog_score,min=0))

        matched_mask = (radgraph_full_labels == 1)
        match_score = feat_orthog_score[matched_mask]
        loss_match = torch.mean(1 - torch.clamp(match_score, min=0)) + loss_orthog
        loss = loss_loc_status + loss_loc_common + loss_dis_status + loss_mach

        loss_dict = {
            'loss':loss
        }
        return loss_dict    
    

    
    def forward(self, 
                image=None, coarse_labels=None, loc_labels=None, full_labels=None, 
                sample_index = None, missing_flag=None, compute_loss:bool=True,
                is_train = True, no_cl= False, exclude_class= False, **kwargs):
                
        B = image.shape[0]
        device = image.device

        if self.disease_book.device != device:
            self.disease_book = self.disease_book.to(device)
        if self.ana_book.device != device:
            self.ana_book = self.ana_book.to(device)


        x = self.image_encoder(image) # [batch_size, patch_num, dim];

        features = x.transpose(0,1) # patch_num b dim


        disease_query_embed = self.disease_embedding_layer(self.disease_book)   # original [disease_num, disease_d];
        disease_query_embed = disease_query_embed.unsqueeze(1).repeat(1, B, 1)   
        loc_query_embed = self.loc_embedding_layer(self.ana_book).unsqueeze(1).repeat(1, B, 1)


        disease_feat, disease_ws = self.dis_decoder(disease_query_embed, features,
                                                    memory_key_padding_mask=None, pos=None, query_pos=None)
        disease_feat = self.dropout_feas(disease_feat)
        loc_feat, loc_ws = self.loc_decoder(loc_query_embed, features,
                                            memory_key_padding_mask=None, pos=None, query_pos=None)
        loc_feat = self.dropout_feas(loc_feat)

        loc_status_pred = self.loc_status_pred(loc_feat)

        if compute_loss:
            loc_common_feat, loc_cosine_sim = self.get_loc_sim(loc_query_embed, loc_feat)
        else:
            loc_common_feat, loc_common_ws = self.loc_common_decoder(loc_query_embed, loc_feat,
                                                  memory_key_padding_mask=None, pos=None, query_pos=None)

        dis_status_pred = self.dis_status_pred(disease_feat)   # [b, disease_num, 2];
        loc_fuse_feat = self.loc_fuse_module(torch.concat([loc_feat, loc_common_feat], dim=-1))

        orthog_score = self.feat_orthogonal_score(loc_fuse_feat, disease_feat)


        if compute_loss:
            if missing_flag.all():
                loss_returned = torch.tensor(0, device=device)
            else:
                dis_num = self.disease_book.shape[0]
                disease_indices = torch.arange(dis_num).view(1, dis_num, 1).expand(B, -1, sample_index.shape[-1])

                loss_returned = self.get_train_loss(
                    loc_status_pred=loc_status_pred[~missing_flag], 
                    loc_labels=loc_labels[~missing_flag],
                    loc_common_feat=loc_common_feat[:, ~missing_flag, :], 
                    loc_sim=loc_cosine_sim[~missing_flag],
                    dis_status_pred=dis_status_pred[~missing_flag], 
                    dis_labels=coarse_labels[~missing_flag],
                    feat_orthog_score=orthog_score[~missing_flag], 
                    radgraph_full_labels=full_labels[~missing_flag]
                )
        else:
            loss_returned = torch.tensor(0, device=device)


        feat_dict = {
            'dis':disease_feat.transpose(0,1),
            'loc':loc_fuse_feat.transpose(0,1),
            'dis_ws':disease_ws,
            'loc_ws':loc_ws
        }

        return {'loss_dict':loss_returned, 'feat_dict':feat_dict}
            




class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, output_size=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.output_size = output_size or hidden_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.output_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj
    

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
