# ------------------------------------------------------------------------
# Blumnet
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# ------------------------------------------------------------------------


import copy
import math
import torch
import torch.nn.functional as F
from torch import nn
from .backbone import build_backbone
from .deformable_transformer import build_deforamble_transformer, inverse_sigmoid
from datasets.sk_points import SkPts
from datasets.data_prefetcher import NestedTensor

skparser = SkPts()


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(context_dim, dim)
        self.v_proj = nn.Linear(context_dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, context):
        # x: [num_layers, batch_size, num_queries, dim]
        # context: [num_layers, batch_size, num_context_queries, dim]
        
        num_layers, batch_size, num_queries, dim = x.shape
        _, _, num_context, _ = context.shape
        
        # Process each decoder layer separately
        output_per_layer = []
        for layer in range(num_layers):
            x_layer = x[layer]  # [batch_size, num_queries, dim]
            context_layer = context[layer]  # [batch_size, num_context, dim]
            
            # Project queries, keys, values
            q = self.q_proj(x_layer).reshape(batch_size, num_queries, self.num_heads, dim // self.num_heads)
            k = self.k_proj(context_layer).reshape(batch_size, num_context, self.num_heads, dim // self.num_heads)
            v = self.v_proj(context_layer).reshape(batch_size, num_context, self.num_heads, dim // self.num_heads)
            
            # Transpose for attention
            q = q.transpose(1, 2)  # [batch_size, num_heads, num_queries, head_dim]
            k = k.transpose(1, 2)  # [batch_size, num_heads, num_context, head_dim]
            v = v.transpose(1, 2)  # [batch_size, num_heads, num_context, head_dim]
            
            # Attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            
            # Get output
            out = (attn @ v).transpose(1, 2).reshape(batch_size, num_queries, dim)
            out = self.out_proj(out)
            
            # Residual connection
            out = out + x_layer
            
            output_per_layer.append(out)
        
        # Stack all layer outputs
        return torch.stack(output_per_layer)


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, cpts=1, out_pts=0):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            cpts: int, number of points to describe a curve
            out_pts, int, the output number of end points and junction points
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.curve_class_embed = nn.Linear(hidden_dim, num_classes)
        self.curve_points_embed = MLP(hidden_dim, hidden_dim, 2 * cpts, 3)
        self.num_feature_levels = num_feature_levels
        # for graph prediction
        assert out_pts >= 0
        self.out_pts = out_pts
        pts_class = 3  # 0-endpts, 1-junctions, 2-nontarget
        self.keypoint_class_embed = nn.Linear(hidden_dim, pts_class)
        self.keypoint_point_direction_embed = MLP(hidden_dim, hidden_dim*2, 4, 6)  # 4: x, y, direction_x, direction_y
        self.query_embed = nn.Embedding(num_queries + out_pts, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.curve_class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.curve_points_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.curve_points_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        self.keypoint_class_embed.bias.data = torch.ones(pts_class) * bias_value
        nn.init.constant_(self.keypoint_point_direction_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.keypoint_point_direction_embed.layers[-1].bias.data, 0)
        num_pred = transformer.decoder.num_layers
        if with_box_refine:
            self.curve_class_embed = _get_clones(self.curve_class_embed, num_pred)
            self.curve_points_embed = _get_clones(self.curve_points_embed, num_pred)
            nn.init.constant_(self.curve_points_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.curve_points_embed
        else:
            nn.init.constant_(self.curve_points_embed.layers[-1].bias.data[2:], -2.0)
            self.curve_class_embed = nn.ModuleList([self.curve_class_embed for _ in range(num_pred)])
            self.curve_points_embed = nn.ModuleList([self.curve_points_embed for _ in range(num_pred)])
            self.keypoint_class_embed = nn.ModuleList([self.keypoint_class_embed for _ in range(num_pred)])
            self.keypoint_point_direction_embed = nn.ModuleList([self.keypoint_point_direction_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        # Cross-attention modules for feature exchange
        self.curve_keypoint_cross_attn = CrossAttention(hidden_dim, hidden_dim)
        self.keypoint_curve_cross_attn = CrossAttention(hidden_dim, hidden_dim)

        # Initialize cross-attention weights
        nn.init.xavier_uniform_(self.curve_keypoint_cross_attn.q_proj.weight)
        nn.init.xavier_uniform_(self.curve_keypoint_cross_attn.k_proj.weight)
        nn.init.xavier_uniform_(self.curve_keypoint_cross_attn.v_proj.weight)
        nn.init.xavier_uniform_(self.curve_keypoint_cross_attn.out_proj.weight)
        nn.init.xavier_uniform_(self.keypoint_curve_cross_attn.q_proj.weight)
        nn.init.xavier_uniform_(self.keypoint_curve_cross_attn.k_proj.weight)
        nn.init.xavier_uniform_(self.keypoint_curve_cross_attn.v_proj.weight)
        nn.init.xavier_uniform_(self.keypoint_curve_cross_attn.out_proj.weight)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        hs, init_reference, inter_references, _, _ = self.transformer(srcs, masks, pos, self.query_embed.weight)
        pts_hs, pts_init_refer, pts_inter_refer = (
            hs[:, :, :self.out_pts], init_reference[:, :self.out_pts], inter_references[:, :, :self.out_pts])
        hs, init_reference, inter_references = (
            hs[:, :, self.out_pts:], init_reference[:, self.out_pts:], inter_references[:, :, self.out_pts:])

        results = {}
        # Extract curve and keypoint features before final prediction
        curve_features = hs
        keypoint_features = pts_hs
        
        # Add cross-attention between features
        curve_enhanced = self.curve_keypoint_cross_attn(curve_features, keypoint_features)
        keypoint_enhanced = self.keypoint_curve_cross_attn(keypoint_features, curve_features)
        
        # Use enhanced features for final prediction
        results['curves'] = self._forward(
            curve_enhanced, init_reference, inter_references, 
            class_embed=self.curve_class_embed, bbox_embed=self.curve_points_embed)
        results['keypoints'] = self._forward(
            keypoint_enhanced, pts_init_refer, pts_inter_refer, 
            class_embed=self.keypoint_class_embed, bbox_embed=self.keypoint_point_direction_embed, direction=True)

        return results

    def _forward(self, hs, init_reference, inter_references, class_embed, bbox_embed, key_prefix='', direction=False):
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = class_embed[lvl](hs[lvl])
            tmp = bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                repeat_num = tmp.shape[-1] // reference.shape[-1]
                tmp = tmp + torch.cat([reference for i in range(repeat_num)], dim=-1)
            if direction:
                outputs_coord_xy = tmp[:, :, :2].sigmoid()
                outputs_coord_direction = tmp[:, :, 2:].sigmoid() * 2 - 1
                outputs_coord = torch.cat([outputs_coord_xy, outputs_coord_direction], dim=-1)
            else:
                outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        out = {f'{key_prefix}pred_logits': outputs_class[-1], f'{key_prefix}pred_points': outputs_coord[-1]}
        if self.aux_loss:
            out[f'{key_prefix}aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, key_prefix=key_prefix)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, key_prefix=''):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{f'{key_prefix}pred_logits': a, f'{key_prefix}pred_points': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_model(config, device):
    num_classes = 2  # [target, non-target]

    backbone = build_backbone(config)

    transformer = build_deforamble_transformer(config)
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=config["num_queries"],
        num_feature_levels=config["num_feature_levels"],
        aux_loss=config["aux_loss"],
        with_box_refine=config["with_box_refine"],
        cpts=config["points_per_path"],
        out_pts=config["out_pts"],
    )
    model.to(device)
    return model
