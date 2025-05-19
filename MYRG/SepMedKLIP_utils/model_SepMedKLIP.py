# modified from https://github.com/tensorflow/models/blob/master/research/slim/nets/s3dg.py
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
'''
args.N
args.d_model
args.res_base_model
args.H 
args.num_queries
args.dropout
args.attribute_set_size
'''

random.seed(7)


class SepMedKLIP(nn.Module):

    def __init__(self, config, ana_book, disease_book, mode='train'):
        super().__init__()

        self.d_model = config['d_model']

        with torch.no_grad():
            bert_model = self._get_bert_basemodel(config['text_encoder'],freeze_layers = None).to(ana_book['input_ids'].device)
            self.ana_book = bert_model(input_ids = ana_book['input_ids'],attention_mask = ana_book['attention_mask'])#(**encoded_inputs)
            self.ana_book = self.ana_book.last_hidden_state[:,0,:]
            self.disease_book = bert_model(input_ids = disease_book['input_ids'][8:,],
                                           attention_mask = disease_book['attention_mask'][8:,])#(**encoded_inputs)
            self.disease_book = self.disease_book.last_hidden_state[:,0,:]

        
        self.disease_embedding_layer = nn.Linear(768,256)
        self.loc_embedding_layer = nn.Linear(768,256)
        
        self.disease_name = [
            'effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation', 'process', 'abnormality', 'enlarge', 'tip', 'low',
            'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air', 'tortuous', 'lead', 'disease', 'calcification', 'prominence',
            'device', 'engorgement', 'picc', 'clip', 'elevation', 'expand', 'nodule', 'wire', 'fluid', 'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
            'hyperinflate', 'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd', 'infiltrate', 'obscure', 'deformity', 'hernia',
            'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline', 'hardware', 'dilation', 'chf', 'redistribution', 'aspiration',
            'tail_abnorm_obs', 'excluded_obs'
        ]
        
        self.excluded_disease = [
            'pneumonia',
            'infiltrate',
            'mass',
            'nodule',
            'emphysema',
            'fibrosis',
            'thicken',
            'hernia'
        ]
        
        self.keep_class_dim = [self.disease_name.index(i) for i in self.disease_name if i not in self.excluded_disease ]
        ''' visual backbone'''
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False)}
        resnet = self._get_res_basemodel(config['res_base_model'])
        num_ftrs = int(resnet.fc.in_features/2)
        self.res_features = nn.Sequential(*list(resnet.children())[:-3])
        self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.res_l2 = nn.Linear(num_ftrs, self.d_model)


        ###################################
        ''' Query Decoder'''
        ###################################

        self.H = config['H'] 
        decoder_layer = TransformerDecoderLayer(self.d_model, config['H'] , 1024,
                                        0.1, 'relu',normalize_before=True)
        
        self.dis_decoder = TransformerDecoder(decoder_layer, 
                                              config['N'] , 
                                              norm=nn.LayerNorm(self.d_model),
                                              return_intermediate=False)
        para_num = sum(p.numel() for p in self.dis_decoder.parameters() if p.requires_grad)
        self.loc_decoder = TransformerDecoder(decoder_layer, 
                                              config['N'] , 
                                              norm=nn.LayerNorm(self.d_model),
                                              return_intermediate=False)
        self.loc_common_decoder = TransformerDecoder(decoder_layer, 
                                              config['N'] , 
                                              norm=nn.LayerNorm(self.d_model),
                                              return_intermediate=False)
        self.fine_pred_loc_decoder = TransformerDecoder(decoder_layer, 
                                              config['N'] , 
                                              norm=nn.LayerNorm(self.d_model),
                                              return_intermediate=False)
        self.fine_pred_dis_decoder = TransformerDecoder(decoder_layer, 
                                              config['N'] , 
                                              norm=nn.LayerNorm(self.d_model),
                                              return_intermediate=False)


        self.dropout_feas = nn.Dropout(config['dropout'] )

        self.dis_classifier = nn.Linear(self.d_model,config['attribute_set_size'])
        self.loc_status_classifier = nn.Linear(self.d_model,config['attribute_set_size'])

        self.apply(self._init_weights)

        assert 'loss_weight' in config, "Missing loss weights."
        self.config = config
        

    def _get_res_basemodel(self, res_model_name):
        try:
            res_model = self.resnet_dict[res_model_name]
            print("Image feature extractor:", res_model_name)
            return res_model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")


    def _get_bert_basemodel(self, bert_model_name, freeze_layers):
        try:
            model = AutoModel.from_pretrained(bert_model_name)#, return_dict=True)
            print("text feature extractor:", bert_model_name)
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
        return model


    def image_encoder(self, xis):
        #patch features
        """
        16 torch.Size([16, 1024, 14, 14])
        torch.Size([16, 196, 1024])
        torch.Size([3136, 1024])
        torch.Size([16, 196, 256])
        """
        batch_size = xis.shape[0]
        res_fea = self.res_features(xis) #batch_size,feature_size,patch_num,patch_num
        res_fea = rearrange(res_fea,'b d n1 n2 -> b (n1 n2) d')
        h = rearrange(res_fea,'b n d -> (b n) d')
        #batch_size,num,feature_size
        # h = h.squeeze()
        x = self.res_l1(h)
        x = F.relu(x)

        x = self.res_l2(x)
        out_emb = rearrange(x,'(b n) d -> b n d',b=batch_size)
        return out_emb
    

    def loc_status_pred(self, loc_feat):
        # lead model to focus on specific location;
        # [loc_num,bz,dim] -> [bz,loc_num,dim] -> [bz,loc_num,2];
        # may also be affected by imbalance;
        loc_status_pred = self.loc_status_classifier(loc_feat.transpose(0,1))
        return loc_status_pred
    

    def get_loc_sim(self, loc_query_embed, loc_feat):
        loc_common_feat, loc_common_ws = self.loc_common_decoder(loc_query_embed, loc_feat,
                                                  memory_key_padding_mask=None, pos=None, query_pos=None)
        rand_idx = random_derangement(loc_common_feat.shape[1])   # along batch dim;
        loc_common_feat_rand_arrange = loc_common_feat[:,rand_idx,:]
        loc_cosine_sim = torch.einsum('bij,bkj -> bik', 
                                      F.normalize(loc_common_feat.transpose(0,1), dim=-1), 
                                      F.normalize(loc_common_feat_rand_arrange.transpose(0,1), dim=-1))
        return loc_common_feat, loc_cosine_sim
    

    def dis_status_pred(self, disease_feat):
        disease_logits = self.dis_classifier(disease_feat.transpose(0,1))
        return disease_logits
    

    def feat_orthogonal_score(self, loc_common_feat, disease_feat):
        orthog_eval_mat = torch.einsum('bij,bkj -> bik', 
                                   F.normalize(loc_common_feat.transpose(0,1), dim=-1), 
                                   F.normalize(disease_feat.transpose(0,1), dim=-1))
        return orthog_eval_mat


    @staticmethod
    def _init_weights(module):
        r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()



def random_derangement(n):
    assert n > 1, f"The random index need use n > 1 but get n: {n}."
    while True:
        v = [i for i in range(n)]
        for j in range(n - 1, -1, -1):
            p = random.randint(0, j)
            if v[p] == j:
                break
            else:
                v[j], v[p] = v[p], v[j]
        else:
            if v[0] != 0:
                return tuple(v)
            
