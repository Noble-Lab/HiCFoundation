
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# AdPE: https://github.com/maple-research-lab/AdPE
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block,PatchEmbed
import numpy as np

from model.pos_embed import get_2d_sincos_pos_embed_rectangle,convert_count_to_pos_embed_cuda
from model.SSIM import ssim 

def apply_symmectric_noise(noise, diag):
    """
    noise: [B,M,N] noise matrix
    diag: [B,1] diagonal position
    """
    B, M, N = noise.shape
    max_size = max(M,N)
    for k in range(B):
        cur_array = noise[k]
        cur_diag = int(diag[k])
        if abs(cur_diag)>=max_size:
            continue
        if cur_diag <0:
            #the diag starts at [diag,0]
            cur_diag = abs(cur_diag)
            row_start = cur_diag
            col_start = 0
            if M-abs(cur_diag)<N:
                row_end = M
                col_end = M-abs(cur_diag)
            else:
                row_end = N+abs(cur_diag)
                col_end = N
        else:
            #the diag starts at [0,diag]
            row_start = 0
            col_start = cur_diag
            if M+cur_diag<N:
                row_end = M
                col_end = M+cur_diag
            else:
                row_end = N-cur_diag
                col_end = N
            #make this region symmetric
        cur_array[row_start:row_end, col_start:col_end] = cur_array[row_start:row_end, col_start:col_end]+cur_array[row_start:row_end, col_start:col_end].T
        noise[k] = cur_array
    return noise
class Models_HiCFoundation(nn.Module):
    """ 
    HiCFoundation:
    Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=(224,224),
                 patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        
        #encoder specification
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer

        #configure positional embedding
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = img_size
        input_row_size = img_size[0]
        input_col_size = img_size[1]
        self.pos_embed_size = (input_row_size // patch_size, input_col_size // patch_size)
        self.patch_embed  = PatchEmbed(self.img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        #configure encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # decoder specification
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------
        self.decoder_count = nn.Linear(decoder_embed_dim, 1, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed= get_2d_sincos_pos_embed_rectangle(self.pos_embed.shape[2], self.pos_embed_size, True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed_rectangle(self.decoder_pos_embed.shape[2], (self.pos_embed_size[0], self.pos_embed_size[1]), False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x,  mask_ratio,diag=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        mask_ratio: float, masking ratio
        diag: [N,1] diagonal position to symmetrical masking, if None, then random masking
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        pos_row_size, pos_col_size = self.pos_embed_size
        
        noise = torch.rand(N, pos_row_size,pos_col_size, device=x.device)
        if diag is not None:
            noise = apply_symmectric_noise(noise, diag)
        noise = noise.view(N,L)
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        # self.len_keep = len_keep
        # self.mask = mask
        return x_masked, mask, ids_restore
    
    def patchify(self, imgs,in_chans=None):
        """
        imgs: (N, 3, H, W)
        x: (N, L, H*W *self.in_chans)
        """
        if in_chans is None:
            in_chans = self.in_chans
        p = self.patch_size
        h = self.pos_embed_size[0]
        w = self.pos_embed_size[1]

        x = imgs.reshape(shape=(imgs.shape[0], in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * in_chans))
        return x
    
    def unpatchify(self, x,in_chans=None):
        """
        x: (N, L, patch_size**2 *self.in_chans)
        """
        if in_chans is None:
            in_chans = self.in_chans
        p = self.patch_size
        h = self.pos_embed_size[0]
        w = self.pos_embed_size[1]
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], in_chans, h * p, w * p))
        return imgs

    def forward_encoder(self, imgs, total_count=None, diag=None,mask_ratio=0.75):
        """
        imgs: [N, 3, H, W]

        total_count: [N, 1] total count of Hi-C, serve as input to predict the submatrix count
        """
        B, C, H, W = imgs.shape
        
        # embed patches
        x = self.patch_embed(imgs)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio, diag)
        if total_count is None:
            #placeholder if total_count is not provided
            total_count = torch.ones(imgs.shape[0]).to(imgs.device)
            total_count = total_count*1000000000
        # gen count embedding
        total_count = torch.log10(total_count)
        count_embed = convert_count_to_pos_embed_cuda(total_count, self.embed_dim)
        count_embed = count_embed.unsqueeze(1)# (N, 1, D)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_tokens, count_embed, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore
    def forward_decoder(self, x, ids_restore):
        """
        x: (N, L, D), sequence of embeddings (including embeddings of cls token and count token)
        return: 

        """
        # embed tokens
        x = self.decoder_embed(x)
        
        num_additional_token = 2 # 1 cls token and 1 count token
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + num_additional_token - x.shape[1], 1)
        x_ = torch.cat([x[:, num_additional_token:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :num_additional_token, :], x_], dim=1)  # append cls token

        # add pos embed
        x[:,num_additional_token:] = x[:,num_additional_token:] + self.decoder_pos_embed

        # we should not add count information to the decoder here, but we can use all-one embedding to distinguish the cls token and count_token
        x[:,1] = x[:,1]+1#all-one embedding for count token

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # remove cls token
        x = x[:, 1:, :]

        # predictor projection
        patch_pred = self.decoder_pred(x[:,1:,:])
        count_pred = self.decoder_count(x[:,0,:])
        

        return count_pred,patch_pred
    def forward_loss(self,imgs,imgs_mask, pred, mask):
        """
        imgs: [N, 3, H, W]
        imgs_mask: [N, 1, H, W] indicate those 0 regions and mask them in target
        pred: [N, L, D], sequence of embeddings
        mask: [N, L], binary mask, 0 is keep, 1 is remove
    
        """
        if self.in_chans==3:
            #unnormalize the image
            imagenet_mean = np.array([0.485, 0.456, 0.406])
            imagenet_std = np.array([0.229, 0.224, 0.225])
            imagenet_mean = torch.tensor(imagenet_mean,device=imgs.device, requires_grad=False)
            imagenet_std = torch.tensor(imagenet_std,device=imgs.device,requires_grad=False)
            imgs_input = imgs
            imgs = torch.einsum("bchw,c->bchw",imgs,imagenet_std)
            imgs = torch.clip((imgs+ imagenet_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) , 0, 1)
        
            pred_image = self.unpatchify(pred)
            pred_image = torch.einsum("bchw,c->bchw",pred_image,imagenet_std)
            pred_image = torch.clip((pred_image+ imagenet_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) , 0, 1)
            target = self.patchify(imgs_input)
            imgs_mask = self.patchify(imgs_mask,1) #N,L,C
        elif self.in_chans==1:
            imgs = imgs*imgs_mask
            target = self.patchify(imgs)
            pred = torch.sigmoid(pred)
            pred_image = self.unpatchify(pred)
            imgs_mask = self.patchify(imgs_mask)
        #make scale to 0-1 to easy comparison of SSIM
        ssim_loss = 1-ssim(pred_image,imgs, data_range=1, size_average=True)
        #calculate patch contrastive loss by cross comparison between query and ground truth
        target = nn.functional.normalize(target, dim=-1)#N,L,C
        pred = nn.functional.normalize(pred, dim=-1)#N,L,C
        pred_logits = torch.einsum('nlc,nkc->nlk', [pred, target]) #N,L,L
        pred_logits = nn.functional.softmax(pred_logits, dim=-1)
        pred_logits = -torch.log(pred_logits)
        pred_logits = torch.diagonal(pred_logits, dim1=1, dim2=2) #N,L, cross entropy multiply label here`
        imgs_mask = imgs_mask.mean(dim=-1)#N,L
        keep_use = imgs_mask>=0.001 #patch_size**2*0.001, roughtly >2 valid pixels
        #keep_use = mask*keep_use 
        # # even unmasked region should also pay attention
        contrastive_loss = (pred_logits*keep_use).sum()/keep_use.sum()
 

        return ssim_loss,contrastive_loss
        
    def forward(self, imgs, imgs_mask, total_count=None, 
                diag=None,mask_ratio=0.75):
        """
        imgs: [N, 3, H, W]
        imgs_mask: [N, 1, H, W] indicate those 0 regions and mask them in target
        total_count: [N, 1] total count of Hi-C, serve as input to predict the submatrix count
        """
        # encoder
        latent, mask, ids_restore = self.forward_encoder(imgs, total_count,diag,mask_ratio)
        # decoder
        count_pred, pred = self.forward_decoder(latent, ids_restore)  # [N, L, embed_dim]
        # calculate loss
        ssim_loss,contrastive_loss = self.forward_loss(imgs,imgs_mask, pred, mask)
        
        #return pred image and mask in 2D for visualization
        pred_image = self.unpatchify(pred)
        mask = mask.unsqueeze(-1).repeat(1, 1, self.patch_size**2 *self.in_chans)  # (N, H*W, p*p*3)
        mask = self.unpatchify(mask)  # 1 is removing, 0 is keeping
        return ssim_loss,contrastive_loss, count_pred, pred_image, mask

def vit_large_patch16(**kwargs):
    model = Models_HiCFoundation(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model