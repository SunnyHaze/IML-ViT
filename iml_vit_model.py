from modules.window_attention_ViT import ViT as window_attention_vit, SimpleFeaturePyramid, LastLevelMaxPool
from modules.decoderhead import PredictHead

import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial

import sys
sys.path.append('./modules')

class iml_vit_model(nn.Module):
    
    def __init__(
        self, 
        # ViT backbone:
        input_size = 1024,
        patch_size = 16,
        vit_pretrain_path = None, # wether to load pretrained weights
        # Simple_feature_pyramid_network:
        fpn_channels = 256,
        fpn_scale_factors = (4.0, 2.0, 1.0, 0.5),
        embed_dim = 768,
        # MLP embedding:
        mlp_embeding_dim = 256,
        # Edge loss:
        edge_lambda = 20,
    ):
        """init iml_vit_model
        # TODO : add more args
        Args:
            encoder_net (nn.Module): encoder network
            featurePyramid_net (nn.Module): feature extractor
            kernel_size_of_edge_generator (int, optional): kernel size for edge generator for 0-1 mapping. Defaults to 5.
            perceptual_loss (str, optional): choose bewteen 'imagenet' and 'mantranet'. Defaults to 'imagenet'.
            perceptual_loss_path (str) : path to the pretrained weights of the perceptual loss. Defaults to None.
        """
        super(iml_vit_model, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        # window attention vit
        self.encoder_net = window_attention_vit(  
            img_size = input_size,
            patch_size=16,
            embed_dim=embed_dim,
            depth=12,
            num_heads=12,
            drop_path_rate=0.1,
            window_size=14,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            window_block_indexes=[
                # 2, 5, 8 11 for global attention
                0,
                1,
                3,
                4,
                6,
                7,
                9,
                10,
            ],
            residual_block_indexes=[],
            use_rel_pos=True,
            out_feature="last_feat",
            )
        self.vit_pretrain_path = vit_pretrain_path
        
        # simple feature pyramid network
        self.featurePyramid_net = SimpleFeaturePyramid(
            in_feature_shape= (1, embed_dim, 256, 256),
            out_channels= fpn_channels,
            scale_factors=fpn_scale_factors,
            top_block=LastLevelMaxPool(),
            norm="LN",    
        )
        # MLP predict head
        self.predict_head = PredictHead(
            feature_channels=[fpn_channels for i in range(5)], embed_dim=mlp_embeding_dim
        )
        # Edge loss hyper-parameters    
        self.BCE_loss = nn.BCEWithLogitsLoss()
        self.edge_lambda = edge_lambda
        
        self.apply(self._init_weights)
        self._mae_init_weights()
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def _mae_init_weights(self):
        # Load MAE pretrained weights for Window Attention ViT encoder
        if self.vit_pretrain_path != None:
            self.encoder_net.load_state_dict(
                torch.load(self.vit_pretrain_path, map_location='cpu')['model'], # BEIT MAE
                strict=False
            )
            print('load pretrained weights from \'{}\'.'.format(self.vit_pretrain_path))
    
    def forward(self, x:torch.Tensor, masks, edge_masks, shape= None):
        x = self.encoder_net(x)
        x = self.featurePyramid_net(x)
        feature_list = []
        for k, v in x.items():
            feature_list.append(v)
        x = self.predict_head(feature_list)
        
        # up-sample to 1024x1024
        mask_pred = F.interpolate(x, size = (self.input_size, self.input_size), mode='bilinear', align_corners=False)
        
        # compute the loss
        # loss = predict_loss + lambda * edge_loss    # default: lambda == 20
        predict_loss = self.BCE_loss(mask_pred, masks)
        edge_loss = F.binary_cross_entropy_with_logits(
            input = mask_pred,
            target= masks, 
            weight = edge_masks
            ) * self.edge_lambda 
        predict_loss += edge_loss
        mask_pred = torch.sigmoid(mask_pred)
        
        return predict_loss, mask_pred, edge_loss


        
    