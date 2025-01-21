from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from model.pos_embed import get_2d_sincos_pos_embed,get_2d_sincos_pos_embed_rectangle

import numpy as np
from typing import Set

class Finetune_Model_Head(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, vit_backbone,task=1,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,pos_embed_size=(1,250)):
        """
        task 0: fine-tuning setting
        task 1: reproducibility analysis
        task 2: loop calling
        task 3: resolution enhancement
        task 4: epigenomic assay prediction
        task 5: scHi-C enhancement
        task 6: embedding analysis for all regions
        task 7: embedding analysis for specified regions
        """

        super().__init__()
        # --------------------------------------------------------------------------
        # HiCFoundation encoder 
        self.vit_backbone = vit_backbone
        self.embed_dim = vit_backbone.embed_dim
        self.task = task

        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer

        num_patches = self.vit_backbone.patch_embed.num_patches
        self.patch_embed = self.vit_backbone.patch_embed
        patch_size = self.vit_backbone.patch_size
        self.in_chans = self.vit_backbone.in_chans

        self.pos_embed_size = pos_embed_size

        # HiCFoundation decoder 
        self.decoder_embed = nn.Linear(self.embed_dim, decoder_embed_dim, bias=True)
        
        
        if self.task==4:
            #for epigenomic assay prediction
            self.decoder_pos_embed_new = nn.Parameter(torch.zeros(1, num_patches , decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding, avoid loading from previous checkpoint
        else:
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        #different name to avoid loading weights error for this
        if self.task==2 or self.task==3 or self.task==5:
            #for loop calling and resolution enhancement and scHi-C enhancement
            self.decoder_map = nn.Linear(decoder_embed_dim, patch_size**2 * 1, bias=True) # decoder to patch
        elif self.task==4:
            output_dim = patch_size
            num_track = 6
            #for epigenomic assay prediction
            self.map_blocks = []
            for k in range(num_track):
                map_block = nn.Linear(decoder_embed_dim, output_dim)
                self.map_blocks.append(map_block)
            self.map_blocks = nn.ModuleList(self.map_blocks)
        elif self.task==0:
            output_dim = patch_size
            self.decoder_map = nn.Linear(decoder_embed_dim, patch_size**2 * 1, bias=True) #map to 2d
            self.map_block = nn.Linear(decoder_embed_dim, output_dim) # map to 1d
        self.num_additional_token = 2 # 1 cls token and 1 count token
        self.initialize_weights()
    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {'pos_embed', 'cls_token', 'dist_token'}
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        
        if self.task==4:
            decoder_pos_embed =get_2d_sincos_pos_embed_rectangle(self.decoder_pos_embed_new.shape[2], self.pos_embed_size, False)
            self.decoder_pos_embed_new.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        else:
            decoder_pos_embed =get_2d_sincos_pos_embed_rectangle(self.decoder_pos_embed.shape[2], self.pos_embed_size, False)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        self._init_weights(self.decoder_embed)
        self._init_weights(self.decoder_blocks)
        self._init_weights(self.decoder_norm)
        if self.task==4:
            for map_block in self.map_blocks:
                self._init_weights(map_block)
        # self._init_weights(self.decoder_map)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        return x
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs

    def unpatchify_channel(self, x,in_chans):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        # h = w = int(x.shape[1]**.5)
        # assert h * w == x.shape[1]
        h= self.pos_embed_size[0]
        w= self.pos_embed_size[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], in_chans, h * p, w * p))
        return imgs
    #@torch.no_grad()
    def forward_backbone(self,img,total_count):
        img=self.vit_backbone.forward_features(img,total_count)
        return img
    def forward_decoder(self,img,total_count=None):
        if total_count is None:
            total_count = torch.ones(img.shape[0]).to(img.device)
            total_count = total_count*1000000000
        x = self.forward_backbone(img,total_count)
        if self.task in [6, 7]:
            embedding_list = []
            embedding_list.append(x)
        # embed tokens
        x = self.decoder_embed(x)
        # add pos embed

        num_additional_token = self.num_additional_token # 1 cls token and 1 count token
        # append mask tokens to sequence
        # add pos embed
        if self.task==4:
            x[:,num_additional_token:] = x[:,num_additional_token:] + self.decoder_pos_embed_new
        else:
            x[:,num_additional_token:] = x[:,num_additional_token:] + self.decoder_pos_embed

        # we should not add count information to the decoder here, but we can use all-one embedding to distinguish the cls token and count_token
        x[:,1] = x[:,1]+1#all-one embedding for count token

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
            if self.task in [6, 7]:
                embedding_list.append(x)
        x = self.decoder_norm(x)
        if self.task in [6, 7]:
            return embedding_list
        else:
            return x 
        
     

    
    def forward(self, img,total_count=None):
        """input hic image"""
        if self.task==0:
            #for fine-tuning
            decoder_output = self.forward_decoder(img,
                                                total_count=total_count)
            submatrix_embedding = decoder_output[:,0,:]
            pred_2d = self.decoder_map(decoder_output)
            pred_2d = pred_2d[:,self.num_additional_token:,:]
            pred_2d = self.unpatchify_channel(pred_2d,1)
            patch_embedding = decoder_output[:,self.num_additional_token:,:]
            num_patch_row = self.pos_embed_size[0]
            num_patch_col = self.pos_embed_size[1]
            pred_1d = patch_embedding.reshape(shape=(patch_embedding.shape[0], num_patch_row,num_patch_col,-1)) #average all columns
            pred_1d = torch.mean(pred_1d,dim=2) #N, H, C
            pred_1d = self.map_block(pred_1d) #N, H, D, where D is the output_dim
            pred_1d = pred_1d.reshape(pred_1d.shape[0],-1) #N, H*D
            return submatrix_embedding, pred_2d[:,0,:], pred_1d
        elif self.task==1:
            #for reproducibility analysis
            decoder_output = self.forward_decoder(img,
                                                total_count=total_count)
            return decoder_output[:,0,:]
        elif self.task==2 or self.task==3 or self.task==5:
            #for loop calling, resolution enhancement and scHi-C enhancement
            decoder_output = self.forward_decoder(img,
                                                total_count=total_count)
            decoder_output = self.decoder_map(decoder_output)
            # use patch-wise token
            decoder_output= decoder_output[:,self.num_additional_token:,:]
            pred_image = self.unpatchify_channel(decoder_output,1)
            return pred_image[:,0,:]
        
        elif self.task==4:
            #for epigenomic assay prediction
            decoder_output = self.forward_decoder(img,
                                                total_count=total_count)
            decoder_output = decoder_output[:,self.num_additional_token:,:]
            num_patch_row = self.pos_embed_size[0]
            num_patch_col = self.pos_embed_size[1]
            x = decoder_output.reshape(shape=(decoder_output.shape[0], num_patch_row,num_patch_col,-1)) #average all columns
            x = torch.mean(x,dim=2) #N, H, C
            #forward invidiual track's small block to map to different track's output
            output = []
            for map_block in self.map_blocks:
                y = x
                y = map_block(y) #N, H, D, where D is the output_dim
                y = y.reshape(y.shape[0],-1) #N, H*D
                output.append(y)
            output = torch.stack(output, dim=1) #change to N, num_track, C
            return output
        
        elif self.task in [6, 7]:
            embedding_list = self.forward_decoder(img,
                                                total_count=total_count)
            #remove cls and additional token, which is not very useful in pre-training
            final_embedding = []
            for embedding in embedding_list:
                embedding = embedding[:,self.num_additional_token:,:]
                # shapre N, L, C
                # reshape to N, H,W,C
                num_patch_row = self.pos_embed_size[0]
                num_patch_col = self.pos_embed_size[1]
                embedding = embedding.reshape(shape=(embedding.shape[0], num_patch_row,num_patch_col,-1)) 
                final_embedding.append(embedding)
            return final_embedding

        else:
            print("Task ",self.task," is not implemented")
            print("Please specify the task using --task with 1,2,3,4,5,6,7")
            raise NotImplementedError(f"Task {self.task} is not implemented")

