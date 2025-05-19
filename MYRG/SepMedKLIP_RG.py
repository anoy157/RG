import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from minigpt4.common.registry import registry
from minigpt4.models.base_model import disabled_train
from minigpt4.models.minigpt_base import MiniGPTBase
from minigpt4.models.Qformer import BertConfig, BertLMHeadModel

import torchvision.models as models
import copy
import torch.nn.functional as F
from typing import Optional, List
from torch import Tensor, einsum
import json
from transformers import AutoTokenizer, AutoModel
from einops import rearrange, repeat
from .SepMedKLIP_utils.model_SepMedKLIP_IM import SepMedKLIP_V5 as SepMedKLIP
from transformers import StoppingCriteria, StoppingCriteriaList
from minigpt4.conversation.conversation import StoppingCriteriaSub
import torch.nn.init as init


@registry.register_model("SepMedKLIP_RG")
class SepMedKLIP_RG(MiniGPTBase):
    """
    MiniGPT-v2 model
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/minigpt_v2.yaml",
    }

    def __init__(
            self,
            vit_model="eva_clip_g",
            img_size=448,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            llama_model="",
            prompt_template='[INST] {} [/INST]',
            max_txt_len=300,
            end_sym='\n',
            lora_r=64,
            lora_target_modules=["q_proj", "v_proj"],
            lora_alpha=16,
            lora_dropout=0.05,
            chat_template=False,
            use_grad_checkpoint_llm=False,
            max_context_len=3800,
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
            fine_encoder_config=None,   # the config for the fine grained visual encoder;
            contrastive_learning:bool=False,  # whether use contrastive learning;
            contrastive_learning_config=None,
            gate_config=None
    ):
        super().__init__(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llama_model=llama_model,
            max_txt_len=max_txt_len,
            max_context_len=max_context_len,
            end_sym=end_sym,
            prompt_template=prompt_template,
            low_resource=low_resource,
            device_8bit=device_8bit,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        # img_f_dim = self.visual_encoder.num_features * 4
        # self.llama_proj = nn.Linear(
        #     img_f_dim, self.llama_model.config.hidden_size
        # )
        self.chat_template = chat_template

        if use_grad_checkpoint_llm:
            self.llama_model.gradient_checkpointing_enable()

        # The fine-grained visual encoder;
        self.fine_grained_encoder = self.get_fine_grained_encoder(fine_encoder_config)
        
        if hasattr(self, 'visual_encoder'):
            del self.visual_encoder
        if hasattr(self, 'ln_vision'):
            del self.ln_vision


        self.contrastive_learning = contrastive_learning
        if self.contrastive_learning:
            print("Using contrastive learning.")
            assert contrastive_learning_config is not None
            self.use_contrastive_learning(contrastive_learning_config)

        self.gate_config = gate_config
        self.gate_projection = nn.Parameter(torch.randn(118, 1, fine_encoder_config['H']))
        self.gate_out_norm = LayerNorm(self.llama_model.config.hidden_size)
        init.kaiming_uniform_(self.gate_projection)

    def get_fine_grained_encoder(self, config):
        logging.info("Initializing fine grained encoder.")
        json_book = json.load(open(config['disease_book'],'r'))
        disease_book = [json_book[i] for i in json_book]
        
        ana_list = ['trachea', 'left_hilar', 'right_hilar', 'hilar_unspec', 'left_pleural',
                'right_pleural', 'pleural_unspec', 'heart_size', 'heart_border', 'left_diaphragm',
                'right_diaphragm', 'diaphragm_unspec', 'retrocardiac', 'lower_left_lobe', 'upper_left_lobe',
                'lower_right_lobe', 'middle_right_lobe', 'upper_right_lobe', 'left_lower_lung', 'left_mid_lung', 'left_upper_lung',
                'left_apical_lung', 'left_lung_unspec', 'right_lower_lung', 'right_mid_lung', 'right_upper_lung', 'right_apical_lung',
                'right_lung_unspec', 'lung_apices', 'lung_bases', 'left_costophrenic', 'right_costophrenic', 'costophrenic_unspec',
                'cardiophrenic_sulcus', 'mediastinal', 'spine', 'clavicle', 'rib', 'stomach', 'right_atrium', 'right_ventricle', 'aorta', 'svc',
                'interstitium', 'parenchymal', 'cavoatrial_junction', 'cardiopulmonary', 'pulmonary', 'lung_volumes', 'unspecified', 'other']

        loc_explain = config.get('loc_explain', False)
        if loc_explain:
            ana_book = []
            loc_book = json.load(open(config['location_book'],'r'))
            for i in ana_list: ana_book.append(loc_book[i])
        else:
            ana_book = ['It is located at ' + i for i in ana_list]
        print(ana_book)

        tokenizer = AutoTokenizer.from_pretrained(config['text_encoder'])

        def get_tokenizer(tokenizer,target_text):
            target_tokenizer = tokenizer(list(target_text), padding='max_length', truncation=True, max_length=128,return_tensors="pt")
            return target_tokenizer

        ana_book_tokenizer = get_tokenizer(tokenizer,ana_book)
        disease_book_tokenizer = get_tokenizer(tokenizer,disease_book)
        model = SepMedKLIP(config, ana_book_tokenizer, disease_book_tokenizer, mode = 'train')

        logging.info(f"Load checkpoint for fine grained encoder from: {config['checkpoint']}")
        checkpoint = torch.load(config['checkpoint']) 

        new_state_dict = {}
        for k, value in checkpoint['model'].items():
            key = k
            if key not in model.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        model.load_state_dict(new_state_dict, strict=False)

        ckpt_keys = set(new_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        for i in ckpt_keys:
            if i not in model_keys:
                print(f"Removed key: {i}")

        self.llama_proj_loc = nn.Sequential(
            LlamaMLP(config['d_model'], 
                     int(config['d_model'] * config['mlp_coefficient']),
                     self.llama_model.config.hidden_size),
            nn.Dropout(config.get('dropout', 0)),
            LayerNorm(self.llama_model.config.hidden_size)
        )

        self.llama_proj_dis = nn.Sequential(
            LlamaMLP(config['d_model'], 
                     int(config['d_model'] * config['mlp_coefficient']),
                     self.llama_model.config.hidden_size),
            nn.Dropout(config.get('dropout', 0)),
            LayerNorm(self.llama_model.config.hidden_size)
        )

        return model          


    def encode_img_fine(self, samples, compute_loss):

        device = samples['image'].device

        with self.maybe_autocast():
            output_dict = self.fine_grained_encoder(compute_loss=compute_loss, **samples)
            loss_dict, feat_dict = output_dict['loss_dict'], output_dict['feat_dict']
            loc_feat, dis_feat = feat_dict['loc'], feat_dict['dis']
            loc_feat = self.llama_proj_loc(loc_feat.to(device)) 
            dis_feat = self.llama_proj_dis(dis_feat.to(device))
            concat_feat = torch.concat([loc_feat, dis_feat], dim=1)
            concat_attn = torch.concat([torch.stack(feat_dict['loc_ws'], dim=-2), torch.stack(feat_dict['dis_ws'], dim=-2)], dim=1)
            concat_feat, uncertain_loss = self.uncertain_gate(concat_feat, concat_attn)
            if isinstance(loss_dict, dict):
                loss_dict['loss'] = loss_dict['loss'] + uncertain_loss
                loss_dict.update({"uncertain_loss":uncertain_loss})
        return concat_feat, loss_dict
    

    def uncertain_gate(self, feat, attn):
        if self.gate_config.get('detach', False):
            attn = attn.detach()
        p = attn / attn.sum(dim=-1, keepdim=True)
        entropy = -torch.sum(p * torch.log(p + 1e-6), dim=-1)
        projected = torch.einsum('ben,ecn->bec', entropy, self.gate_projection)  # [b, entity_num, 1]
        gate = torch.sigmoid(projected)
        return_feat = gate * feat
        return_feat = self.gate_out_norm(return_feat)
        e_loss = entropy.mean()
        return return_feat, e_loss


    def encode_img(self, samples, fine_loss:bool=True):
        image = samples['image']
        device = image.device

        if len(image.shape) > 4:
            image = image.reshape(-1, *image.shape[-3:])

        with self.maybe_autocast():
            inputs_llama, fine_loss = self.encode_img_fine(samples, compute_loss=fine_loss)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama, fine_loss



    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        low_resource = cfg.get("low_resource", False)

        prompt_template = cfg.get("prompt_template", '[INST] {} [/INST]')
        max_txt_len = cfg.get("max_txt_len", 300)
        end_sym = cfg.get("end_sym", '\n')

        lora_r = cfg.get("lora_r", 64)
        lora_alpha = cfg.get("lora_alpha", 16)
        lora_dropout = cfg.get("lora_dropout", 0.05)
        chat_template = cfg.get("chat_template", False)

        use_grad_checkpoint_llm = cfg.get("use_grad_checkpoint_llm", False)
        max_context_len = cfg.get("max_context_len", 3800)

        # get the config for the fine grained encoder;
        fine_encoder_config = cfg.get("fine_encoder", None)


        gate_config = cfg.get("gate", None)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llama_model=llama_model,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            low_resource=low_resource,
            end_sym=end_sym,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            chat_template=chat_template,
            use_grad_checkpoint_llm=use_grad_checkpoint_llm,
            max_context_len=max_context_len,
            fine_encoder_config=fine_encoder_config,
            gate_config=gate_config
        )

        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load Minigpt-4-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")


            # Compare keys: ensure all modified keys in the checkpoint exist in the model
            ckpt_keys = set(ckpt['model'].keys())
            model_keys = set(model.state_dict().keys())
            for i in ckpt_keys:
                if i not in model_keys:
                    print(f"Missing key: {i}")
                else:
                    print(f"Find useful key:{i}")

            msg = model.load_state_dict(ckpt['model'], strict=False)

        if fine_encoder_config.get('only_text_embedding_trainable', False):
            # Freeze all parameters
            for param in model.parameters():
                param.requires_grad = False

            trainable_module = [
                'fine_grained_encoder.disease_embedding_layer',
               'fine_grained_encoder.loc_embedding_layer'
            ]

            # Unfreeze the specified layers
            for name, param in model.named_parameters():
                for i in trainable_module:
                    if i in name:
                        param.requires_grad = True

            # Check if freezing was successful
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"{name}: requires_grad={param.requires_grad}")

        return model
    

    def preparing_embedding(self, samples):
        ### prepare input tokens
        if 'image' in samples:
            img_embeds, img_atts, fine_loss = self.encode_img(samples)
        else:
            img_embeds = img_atts = None

        if 'conv_q' in samples:
            # handeling conversation datasets
            conv_q, conv_a = samples['conv_q'], samples['conv_a']

            connect_sym = samples['connect_sym'][0]
            conv_q = [q.split(connect_sym)for q in conv_q]
            conv_a = [a.split(connect_sym) for a in conv_a]

            conv_q = [[self.prompt_template.format(item) for item in items] for items in conv_q]

            cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts, [q[0] for q in conv_q])
            regress_token_ids, regress_atts, part_targets = self.tokenize_conversation(conv_q, conv_a)

        else:
            if "instruction_input" in samples:
                instruction = samples["instruction_input"]
            elif self.prompt_list:
                instruction = random.choice(self.prompt_list)
            else:
                instruction = None

            if hasattr(self, 'chat_template') and self.chat_template:
                instruction = [self.prompt_template.format(instruct) for instruct in instruction]

            if 'length' in samples:
                # the input is a image train (like videos)
                bsz, pn, hs = img_embeds.shape
                img_embeds = img_embeds.reshape(len(samples['image']), -1, pn, hs)
                cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts, instruction, samples['length'])
            else:
                cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts, instruction)

            ### prepare target tokens
            self.llama_tokenizer.padding_side = "right"
            text = [t + self.end_sym for t in samples["answer"]]

            regress_tokens = self.llama_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(self.device)

            regress_token_ids = regress_tokens.input_ids
            regress_atts = regress_tokens.attention_mask
            part_targets = regress_token_ids.masked_fill(
                regress_token_ids == self.llama_tokenizer.pad_token_id, -100
            )

        regress_embeds = self.embed_tokens(regress_token_ids)

        if self.contrastive_learning: 
            return cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets, fine_loss, img_embeds
        
        return cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets, fine_loss


    def forward(self, samples, reduction='mean'):
        if not self.contrastive_learning:
            cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets, fine_loss = \
                self.preparing_embedding(samples)
        else:
            cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets, fine_loss, img_embeds = \
                self.preparing_embedding(samples)            

        inputs_embeds, attention_mask, input_lens = \
            self.concat_emb_input_output(cond_embeds, cond_atts, regress_embeds, regress_atts)

        bos = torch.ones_like(part_targets[:, :1]) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        bos_atts = cond_atts[:, :1]

        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([bos_atts, attention_mask], dim=1)

        # ensemble the final targets
        targets = torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
                             dtype=torch.long).to(self.device).fill_(-100)

        for i, target in enumerate(part_targets):
            targets[i, input_lens[i]+1:input_lens[i]+len(target)+1] = target  # plus 1 for bos

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                reduction=reduction
            )
        loss = outputs.loss

        if isinstance(fine_loss, dict):
            total_loss = loss + fine_loss['loss']
        else:
            total_loss = loss

        return {"llama_loss": loss, "fine_loss":fine_loss, "loss":total_loss}
    

    @torch.no_grad()
    def generate(
        self,
        # images,
        # texts,
        samples,
        num_beams=1,
        max_new_tokens=20,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1,
        length_penalty=1,
        temperature=1,
        do_sample=False,
        stop_words_ids=[2],
    ):
        '''
            function for generate test use
        '''

        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
            stops=[torch.tensor([i]).to(self.device) for i in stop_words_ids])])

        samples['image'] = samples['image'].to(self.device)
        texts = samples['instruction_input']

        img_embeds, atts_img, fine_loss = self.encode_img(samples, fine_loss=False)
        image_lists = [[image_emb[None]] for image_emb in img_embeds]

        batch_embs = [self.get_context_emb(text, img_list) for text, img_list in zip(texts, image_lists)]

        batch_size = len(batch_embs)
        max_len = max([emb.shape[1] for emb in batch_embs])
        emb_dim = batch_embs[0].shape[2]
        dtype = batch_embs[0].dtype
        device = batch_embs[0].device

        embs = torch.zeros([batch_size, max_len, emb_dim], dtype=dtype, device=device)
        attn_mask = torch.zeros([batch_size, max_len], dtype=torch.int, device=device)
        for i, emb in enumerate(batch_embs):
            emb_len = emb.shape[1]
            embs[i, -emb_len:] = emb[0]
            attn_mask[i, -emb_len:] = 1

        with self.maybe_autocast():
            outputs = self.llama_model.generate(
                inputs_embeds=embs,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                temperature=temperature,
                do_sample=do_sample,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                # stopping_criteria=stopping_criteria,
            )

        answers = []
        for output_token in outputs:
            if output_token[0] == 0:
                output_token = output_token[1:]
            output_texts = self.llama_tokenizer.decode(output_token, skip_special_tokens=True)
            output_texts = output_texts.split('</s>')[0]  # remove the stop sign </s>
            output_texts = output_texts.replace("<s>", "")
            output_texts = output_texts.split(r'[/INST]')[-1].strip()
            answers.append(output_texts)

        return answers
    

    @torch.no_grad()
    def custom_generate(
        self,
        # images,
        # texts,
        # num_beams=1,
        # max_new_tokens=20,
        # min_length=1,
        # top_p=0.9,
        # repetition_penalty=1,
        # length_penalty=1,
        # temperature=1,
        # do_sample=False,
        samples,
        stop_words_ids=[2],
        gerneration_config_dict:dict=None
    ):
        '''
            function for generate test use
        '''

        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
            stops=[torch.tensor([i]).to(self.device) for i in stop_words_ids])])
        
        samples['image'] = samples['image'].to(self.device)
        texts = samples['instruction_input']

        img_embeds, atts_img, fine_loss = self.encode_img(samples, fine_loss=False)
        image_lists = [[image_emb[None]] for image_emb in img_embeds]

        batch_embs = [self.get_context_emb(text, img_list) for text, img_list in zip(texts, image_lists)]

        batch_size = len(batch_embs)
        max_len = max([emb.shape[1] for emb in batch_embs])
        emb_dim = batch_embs[0].shape[2]
        dtype = batch_embs[0].dtype
        device = batch_embs[0].device

        embs = torch.zeros([batch_size, max_len, emb_dim], dtype=dtype, device=device)
        attn_mask = torch.zeros([batch_size, max_len], dtype=torch.int, device=device)
        for i, emb in enumerate(batch_embs):
            emb_len = emb.shape[1]
            embs[i, -emb_len:] = emb[0]
            attn_mask[i, -emb_len:] = 1

        with self.maybe_autocast():
            outputs = self.llama_model.generate(
                inputs_embeds=embs,
                attention_mask=attn_mask,
                **gerneration_config_dict
            )


        answers = []
        for output_token in outputs:
            if output_token[0] == 0:
                output_token = output_token[1:]
            output_texts = self.llama_tokenizer.decode(output_token, skip_special_tokens=True)
            output_texts = output_texts.split('</s>')[0]  # remove the stop sign </s>
            output_texts = output_texts.replace("<s>", "")
            output_texts = output_texts.split(r'[/INST]')[-1].strip()
            answers.append(output_texts)

        return answers
    


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