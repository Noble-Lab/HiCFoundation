"""
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# AdPE: https://github.com/maple-research-lab/AdPE
# --------------------------------------------------------
"""
from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed
import numpy as np

from model_funcs import get_2d_sincos_pos_embed_rectangle, convert_count_to_pos_embed_cuda
from SSIM import ssim


def apply_symmectric_noise(noise, diag, patch_size):
    """
    In this function, we find the diagonal coordinates
    and make the square region surrounding the diagonal
    symmetric.
    noise:
    diag:
    :param noise: np.array, [B,M,N] noise matrix
    :param diag: np.array, [B,1] diagonal position
    :param patch_size: int, patch size
    :return: np.array, [B,M,N] the symmetric noise matrix
    """
    # Get three dimensions of the noise array
    B, M, N = noise.shape

    # Extract the largest dimension of the noise array
    max_size = max(M, N)

    # For every index k in the B dimension
    for k in range(B):
        # select the current noise array
        cur_array = noise[k]
        cur_diag = int(diag[k])
        if abs(cur_diag)>=max_size:
            continue
        # if diagonal starts in the first column.
        if cur_diag < 0:
            # the diag starts at [diag,0]
            cur_diag = abs(cur_diag)
            # diagonal start coordinates
            row_start = cur_diag
            col_start = 0
            # If the diagonal row ends at the patch end
            # and the column ends in the upper triangle.
            if M - abs(cur_diag) < N:
                row_end = M
                col_end = M - abs(cur_diag)
            # If the diagonal column ends at the patch end
            # and the row ends in the upper triangle.
            else:
                row_end = N + abs(cur_diag)
                col_end = N
        # If the diagonal starts in the first row.
        else:
            # the diagonal starts at [0,diag]
            # diagonal start coordinates
            row_start = 0
            col_start = cur_diag

            # If the diagonal row ends at the patch end
            # and the column ends in the lower triangle.
            if M + cur_diag < N:
                row_end = M
                col_end = M + cur_diag
            # If the diagonal column ends at the patch end
            # and the row ends in the lower triangle.
            else:
                row_end = N - cur_diag
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

    def __init__(self, img_size=(224, 224),
                 patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        """
        Constructor function for the Hi-C foundation model
        :param img_size: tuple, Hi-C submatrix size.
        :param patch_size: int, Hi-C patch size.
        :param in_chans: int, number of input channels.
        :param embed_dim: int, number of encoder embedding dimensions.
        :param depth: int, number of transformer layers in encoder.
        :param num_heads: int, number of attention heads in each
                            transformer layer in the encoder.
        :param decoder_embed_dim: int, number of decoder embedding
                                    dimensions.
        :param decoder_depth: int, number of transformer layers in
                                    the decoder.
        :param decoder_num_heads: int, number of attention heads in
                                    each transformer layer in the decoder.
        :param mlp_ratio: float, defines how much larger the hidden dimension
                                 of the MLP in vision transformer is compared
                                 to the embedding dimension.
        :param norm_layer: torch.nn, which norm to apply to the layers.
        """
        super().__init__()

        # encoder specification
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer

        # configure positional embedding
        # specify the submatrix size operator
        # as a tuple.
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = img_size

        # specify rows and columns size
        input_row_size = img_size[0]
        input_col_size = img_size[1]

        # Divide input size by patch size to get the number of positional embeddings
        self.pos_embed_size = (input_row_size // patch_size, input_col_size // patch_size)

        # Call the patch embed function to generate patch embeddings
        self.patch_embed = PatchEmbed(self.img_size, patch_size, in_chans, embed_dim)

        # Get number of patches from the patch embedding output
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        #configure encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        # Get the vision transformer design blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        # Generate the layer norms for the encoder embedding dimensions
        self.norm = norm_layer(embed_dim)

        # configure decoder
        # Linear decoder layer
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        # Mask token parameter
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # Define decoder positional embeddings
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding
        # Define the vision transformer style decoder blocks
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        # Generate the layer norms for the decoder embedding dimensions
        self.decoder_norm = norm_layer(decoder_embed_dim)
        # Decoder prediction linear layer
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)
        # Count prediction layer
        self.decoder_count = nn.Linear(decoder_embed_dim, 1, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize the weights of the encoder and decoder networks
        using xavier_uniform and the class and mask tokens
        using truncated normal with standard deviation of 0.02.

        :return: None.
        """
        # Get encoder positional embedding
        pos_embed = get_2d_sincos_pos_embed_rectangle(self.pos_embed.shape[2], self.pos_embed_size, True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # Get decoder positional embedding
        decoder_pos_embed = get_2d_sincos_pos_embed_rectangle(self.decoder_pos_embed.shape[2],
                                                              (self.pos_embed_size[0], self.pos_embed_size[1]), False)
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
        """
        Initialize linear layers with xavier uniform and
        layer norms with constant values.
        :param m: nn.Linear or nn.LayerNorm
        :return: None.
        """
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio, diag=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        :param x: torch.tensor, [N, L, D], sequence
        :param mask_ratio: float, masking ratio
        :param diag: torch.tensor, [N,1] diagonal position to symmetrical masking, if None, then random masking
        :return: x_masked, mask, ids_restore
        """
        # batch, length, dim
        N, L, D = x.shape
        # Unmasked region ratio
        len_keep = int(L * (1 - mask_ratio))
        pos_row_size, pos_col_size = self.pos_embed_size

        # Generate random noise
        noise = torch.rand(N, pos_row_size, pos_col_size, device=x.device)

        # If the patch contains diagonal, make the noise symmetric
        if diag is not None:
            noise = apply_symmectric_noise(noise, diag, self.patch_size)
        noise = noise.view(N, L)

        # sort noise for each sample
        # ascend: small is kept, large is removed
        ids_shuffle = torch.argsort(noise, dim=1)
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

    def patchify(self, imgs, in_chans=None):
        """
        Convert data into patches with RGB channels.
        :param imgs: torch.tensor, (N, 3, H, W)
        :param in_chans: int, number of in channels
        :return: x, (imgs.shape[0], h * w, p ** 2 * in_chans)
        """

        if in_chans is None:
            in_chans = self.in_chans
        p = self.patch_size
        h = self.pos_embed_size[0]
        w = self.pos_embed_size[1]

        x = imgs.reshape(shape=(imgs.shape[0], in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * in_chans))
        return x

    def unpatchify(self, x, in_chans=None):
        """
        Convert patches back to raw data format.
        :param x: (N, L, patch_size**2 *self.in_chans)
        :param in_chans: int, number of
        :return: imgs
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
        Forward function for the encoder.
        :param imgs: torch.tensor, input patch
        :param species_id: torch.tensor, species ID.
        :param total_count: torch.tensor, [N, 1] total count of Hi-C, serves as input to predict the submatrix count
        :param diag: torch.tensor, position of the diagonal in the input patch
        :param mask_ratio: float, how much of the input patch are we masking.
        :return: x, mask, id_restore
        """
        B, C, H, W = imgs.shape

        # embed patches
        x = self.patch_embed(imgs)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio, diag)
        if total_count is None:
            # placeholder if total_count is not provided
            total_count = torch.ones(imgs.shape[0]).to(imgs.device)
            total_count = total_count * 1000000000
        # gen count embedding
        total_count = torch.log10(total_count)
        count_embed = convert_count_to_pos_embed_cuda(total_count, self.embed_dim)
        count_embed = count_embed.unsqueeze(1)  # (N, 1, D)
        # species_id: (N,) tensor with values in [0, 11]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_tokens, count_embed, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """
        Forward function for the decoder.
        :param x: torch.tensor, input
        :param ids_restore: torch.tensor, unmasked regions
        :return: count_pred, patch_pred
        """
        # embed tokens
        x = self.decoder_embed(x)
        # one cls token and one count token
        num_additional_token = 2
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + num_additional_token - x.shape[1], 1)
        x_ = torch.cat([x[:, num_additional_token:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :num_additional_token, :], x_], dim=1)  # append cls token

        # add pos embed
        x[:, num_additional_token:] = x[:, num_additional_token:] + self.decoder_pos_embed

        # we should not add count information to the decoder here,
        # but we can use all-one embedding to distinguish the cls token and count_token
        x[:, 1] = x[:, 1] + 1  # all-one embedding for count token

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # remove cls token
        x = x[:, 1:, :]

        # predictor projection
        patch_pred = self.decoder_pred(x[:, 1:, :])
        count_pred = self.decoder_count(x[:, 0, :])

        return count_pred, patch_pred

    def forward_loss(self, imgs, imgs_mask, pred, mask):
        """
        Compute all the losses
        :param imgs: torch.tensor, input
        :param imgs_mask: torch.tensor, input mask
        :param pred: torch.tensor, prediction
        :param mask: Unused
        :return: ssim_loss, contrastive_loss
        """
        # If the data is in RGB format.
        if self.in_chans == 3:
            # unnormalize the image
            imagenet_mean = np.array([0.485, 0.456, 0.406])
            imagenet_std = np.array([0.229, 0.224, 0.225])
            imagenet_mean = torch.tensor(imagenet_mean, device=imgs.device, requires_grad=False)
            imagenet_std = torch.tensor(imagenet_std, device=imgs.device, requires_grad=False)
            imgs_input = imgs
            imgs = torch.einsum("bchw,c->bchw", imgs, imagenet_std)
            imgs = torch.clip((imgs + imagenet_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)), 0, 1)

            pred_image = self.unpatchify(pred)
            pred_image = torch.einsum("bchw,c->bchw", pred_image, imagenet_std)
            pred_image = (pred_image + imagenet_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
            pred_image = torch.clip(pred_image, 0, 1)
            target = self.patchify(imgs_input)
            imgs_mask = self.patchify(imgs_mask, 1)  # N,L,C
        # If the data is not in RGB format.
        elif self.in_chans == 1:
            imgs = imgs * imgs_mask
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
        pred_logits = torch.diagonal(pred_logits, dim1=1, dim2=2)  # N,L, cross entropy multiply label here`
        imgs_mask = imgs_mask.mean(dim=-1)  # N,L
        keep_use = imgs_mask >= 0.001  # patch_size**2*0.001, roughtly >2 valid pixels
        # keep_use = mask*keep_use
        # # even unmasked region should also pay attention
        contrastive_loss = (pred_logits * keep_use).sum() / keep_use.sum()

        return ssim_loss,contrastive_loss
        
    def forward(self, imgs, imgs_mask, total_count=None, 
                diag=None,mask_ratio=0.75, if_train=True):
        """
        Forward function for Hi-C foundation.
        :param imgs: torch.tensor, input
        :param imgs_mask: torch.tensor, input mask
        :param species_id: int, species ID
        :param total_count: float, total read coverage
        :param diag: torch.tensor, location of the diagonal
        :param mask_ratio: float, masking ratio
        :param if_train: bool, if training
        :return: ssim_loss, contrastive_loss, count_pred, pred_image, mask
        """

        # encoder
        latent, mask, ids_restore = self.forward_encoder(imgs, total_count,diag,mask_ratio)
        # decoder
        count_pred, pred = self.forward_decoder(latent, ids_restore)  # [N, L, embed_dim]
        # calculate loss
        ssim_loss, contrastive_loss = self.forward_loss(imgs, imgs_mask, pred, mask)

        # return pred image and mask in 2D for visualization
        pred_image = self.unpatchify(pred)
        mask = mask.unsqueeze(-1).repeat(1, 1, self.patch_size ** 2 * self.in_chans)  # (N, H*W, p*p*3)
        mask = self.unpatchify(mask)  # 1 is removing, 0 is keeping
        return ssim_loss, contrastive_loss, count_pred, pred_image, mask


def vit_large_patch16(**kwargs):
    """
    Instantiate the Hi-C foundation model.
    :param kwargs: model parameters
    :return: HiCFoundation Object, model
    """
    model = Models_HiCFoundation(
        embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
