#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import math
import re
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random
import os
from .multimodal_encoder.siglip_encoder import SigLipVisionAbstract


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        self.meta_config = config
        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            # self.multi_tower = ...
            # self.text_tower = SigLipTextTower(config.mm_vision_tower)
            self.vision_abstract = None
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def get_text_tower(self):
        text_tower = getattr(self, "text_tower", None)
        # text_tower.load_pretrained()
        return text_tower
    
    def get_vision_abstract(self):
        if self.vision_abstract is None:
            self.vision_abstract = SigLipVisionAbstract(self.meta_config.mm_vision_tower)
        return self.vision_abstract

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        
        if not hasattr(self.config, 'add_faster_video'):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        # self.config.mm_spatial_pool_mode = 'bilinear'
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature
    
    
    def get_2dPool_index(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        need_unsqueeze = (image_feature.dim() == 2)
        if need_unsqueeze:
            image_feature = image_feature.unsqueeze(2) # [B, N, 1]
        
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        
        image_feature = image_feature.float()
        
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        # self.config.mm_spatial_pool_mode = 'bilinear'
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        
        image_feature = (image_feature >= 0.5)
        
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        image_feature = image_feature.squeeze(dim=2)
        
        return image_feature

    def token_merging(self, image_features, index_mask, scaling=1):
        """
        Merges non-retained tokens with their nearest retained tokens based on cosine similarity.

        Args:
            image_features (Tensor): Tensor of shape (B, N, D) where B is the batch size,
                                    N is the number of tokens, and D is the feature dimension.
            index_mask (Tensor): Binary mask of shape (B, N), where `True` means the token is retained,
                                and `False` means the token is not retained.

        Returns:
            merged_features (Tensor): Tensor of shape (B, N, D) where N is the number of tokens
                                    and D is the feature dimension. The merged features are
                                    the average of the retained token and the non-retained tokens.
        """
        B, N, D = image_features.shape
        T = index_mask.sum(dim=1)  # Number of retained tokens for each batch

        # Initialize the merged_features tensor
        merged_features = []
        
        # Use boolean indexing to directly select retained and non-retained tokens
        retained_tokens = []
        non_retained_tokens = []

        for b in range(B):
            retained_tokens.append(image_features[b][index_mask[b]])  # Select tokens where mask is True
            non_retained_tokens.append(image_features[b][~index_mask[b]])

        # Stack them into tensors
        retained_tokens = torch.stack(retained_tokens, dim=0)  # (B, T, D)
        non_retained_tokens = torch.stack(non_retained_tokens, dim=0)  # (B, N - T, D)
        
        # import pdb;pdb.set_trace()
        
        if non_retained_tokens.shape[1] == 0:
            return image_features

        cosine_sim = F.cosine_similarity(non_retained_tokens.unsqueeze(2), retained_tokens.unsqueeze(1), dim=3)
        nearest_token_indices = cosine_sim.argmax(dim=2)  # (B, N - T)
        # Track how many non-retained tokens merge with each retained token
        merge_count = torch.zeros(B, T[0], device=image_features.device, dtype=torch.int)
        # Merge tokens by averaging
        merged_features = torch.zeros_like(retained_tokens)  # (B, T, D)
        merged_features += retained_tokens * scaling
        # Process each non-retained token and add it to its nearest retained token
        expanded_indices = nearest_token_indices  # Shape: [B, N - T]
        merged_features.scatter_add_(1, nearest_token_indices.unsqueeze(-1).expand(-1, -1, D), non_retained_tokens)
        merge_count.scatter_add_(1, expanded_indices, torch.ones_like(expanded_indices, dtype=merge_count.dtype))
        # print(merge_count)

        # Normalize the retained tokens by the number of non-retained tokens merging with them
        merged_features /= (scaling + merge_count.unsqueeze(2))
        
        for b in range(B):
            # Replace the non-retained tokens with the merged features
            image_features[b, index_mask[b]] = merged_features[b]
        
        return image_features
    
    def Sinkhorn_v1(self, K, u, v, max_iter):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

        return T

    def time_block_cuda(self, wdist, eps, xx, yy, iters=50):
        assert wdist.is_cuda, "need GPU tensos"
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start.record()
        with torch.no_grad():
            KK = torch.exp(-wdist / eps)
            T = self.Sinkhorn_v1(KK, xx, yy, iters)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)  # ms
        return ms, T
    
    

    def window_cls_selection(self, image_attentions, visual_token_num, window_size=6):
        '''
        Input: 
            image_attentions: Tensor of shape (B, N), where N is typically 576 (24x24)
            visual_token_num: int, T - total number of tokens to select
            window_size: int, size of square window (default: 6)

        Output:
            token_indices: Tensor of shape (B, T), indices of top tokens selected from each window

        Description:
            Reshape image_attentions to (B, 24, 24), divide into non-overlapping windows of size window_size x window_size,
            select top-k tokens with highest attention from each window, where k = T // num_windows
        '''
        # import pdb;pdb.set_trace()
        
        B, N = image_attentions.shape
        assert N == 24 * 24, "image_attentions must be of shape (B, 576)"
        H = W = 24

        # Reshape to (B, H, W)
        attn_map = image_attentions.view(B, H, W)

        # Calculate number of windows per dimension
        num_windows_h = H // window_size
        num_windows_w = W // window_size
        total_windows = num_windows_h * num_windows_w

        k = visual_token_num // total_windows  # tokens per window
        token_indices = []

        for b in range(B):
            indices_b = []
            for i in range(num_windows_h):
                for j in range(num_windows_w):
                    # Extract window
                    window = attn_map[b, i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size]
                    # Flatten the window
                    window_flat = window.reshape(-1)
                    # Get top-k indices within the window
                    topk_values, topk_indices = torch.topk(window_flat, k)
                    # Map local window indices to global indices in (24x24)
                    for idx in topk_indices:
                        dy, dx = divmod(idx.item(), window_size)
                        global_y = i * window_size + dy
                        global_x = j * window_size + dx
                        global_index = global_y * W + global_x
                        indices_b.append(global_index)
            token_indices.append(indices_b)
            
        token_indices = torch.tensor(token_indices, device=image_attentions.device)

        return token_indices


    def window_cls_selection_siglip(self, image_attentions, visual_token_num, window_size=9):
        '''
        Input: 
            image_attentions: Tensor of shape (B, N), where N is typically 729 (27x27)
            visual_token_num: int, T - total number of tokens to select
            window_size: int, size of square window (default: 9)

        Output:
            token_indices: Tensor of shape (B, T), indices of top tokens selected from each window

        Description:
            Reshape image_attentions to (B, 27, 27), divide into non-overlapping windows of size window_size x window_size,
            select top-k tokens with highest attention from each window, where k = T // num_windows
        '''
        B, N = image_attentions.shape
        assert N == 27 * 27, "image_attentions must be of shape (B, 729)"
        H = W = 27

        # Reshape to (B, H, W)
        attn_map = image_attentions.view(B, H, W)

        # Calculate number of windows per dimension
        num_windows_h = H // window_size
        num_windows_w = W // window_size
        total_windows = num_windows_h * num_windows_w

        k = visual_token_num // total_windows  # tokens per window
        token_indices = []

        for b in range(B):
            indices_b = []
            for i in range(num_windows_h):
                for j in range(num_windows_w):
                    # Extract window
                    window = attn_map[b, i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size]
                    # Flatten the window
                    window_flat = window.reshape(-1)
                    # Get top-k indices within the window
                    topk_values, topk_indices = torch.topk(window_flat, k)
                    # Map local window indices to global indices in (24x24)
                    for idx in topk_indices:
                        dy, dx = divmod(idx.item(), window_size)
                        global_y = i * window_size + dy
                        global_x = j * window_size + dx
                        global_index = global_y * W + global_x
                        indices_b.append(global_index)
            token_indices.append(indices_b)
            
        token_indices = torch.tensor(token_indices, device=image_attentions.device)
        
        # if B == 1:
        #     token_indices = token_indices.unsqueeze(0)

        return token_indices

    
    
    def token_merging_siglip_with_OTA(self, image_features, index_mask, scaling=1):
        """
        Merges non-retained tokens with their nearest retained tokens based on cosine similarity.

        Args:
            image_features (Tensor): Tensor of shape (B, N, D) where B is the batch size,
                                    N is the number of tokens, and D is the feature dimension.
            index_mask (Tensor): Binary mask of shape (B, N), where `True` means the token is retained,
                                and `False` means the token is not retained.

        Returns:
            merged_features (Tensor): Tensor of shape (B, N, D) where N is the number of tokens
                                    and D is the feature dimension. The merged features are
                                    the average of the retained token and the non-retained tokens.
        """
        B, N, D = image_features.shape
        T_index = index_mask.sum(dim=1)  # Number of retained tokens for each batch

        H = W = int(image_features.size(1) ** 0.5)
        assert H * W == image_features.size(1), "rectangualr iamge tokens"
        
        # Initialize the merged_features tensor
        merged_features = []
        
        # Use boolean indexing to directly select retained and non-retained tokens
        retained_tokens = []
        non_retained_tokens = []

        for b in range(B):
            retained_tokens.append(image_features[b][index_mask[b]])  # Select tokens where mask is True
            non_retained_tokens.append(image_features[b][~index_mask[b]])

        # Stack them into tensors
        retained_tokens = torch.stack(retained_tokens, dim=0)  # (B, T, D)
        non_retained_tokens = torch.stack(non_retained_tokens, dim=0)  # (B, N - T, D)
        
        # import pdb;pdb.set_trace()
        
        # (Ntot,2): (x,y) in [0,1]
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, H, device=image_features.device, dtype=image_features.dtype),
            torch.linspace(0, 1, W, device=image_features.device, dtype=image_features.dtype),
            indexing='ij'
        )
        
        # (B,Ntot,2)
        pos_grid = torch.stack([xx, yy], dim=-1).view(-1, 2).unsqueeze(0).repeat(image_features.size(0), 1, 1)
        
        # (B, N_r, 2)
        pos_ret = torch.stack([pos_grid[b][index_mask[b]] for b in range(B)], dim=0)
        # (B, N_nr, 2)
        pos_nr = torch.stack([pos_grid[b][~index_mask[b]] for b in range(B)], dim=0)
        
        # (B, M, N)
        wdist_spat = torch.cdist(pos_nr, pos_ret, p=2)
        # wdist_spat = torch.cdist(pos_nr, pos_ret, p=2)**2
        
        if non_retained_tokens.shape[1] == 0:
            return image_features

        # # A: [N, D], B: [M, D]
        # A_n = torch.nn.functional.normalize(retained_tokens, dim=-1)
        # B_n = torch.nn.functional.normalize(non_retained_tokens, dim=-1)
        # # 直接 GEMM，不会创建 [N,M,D] 中间量
        # cosine_sim = torch.bmm(A_n, B_n.transpose(1, 2))


        # implement this to aviod OOM.
        cosine_sim = []
        for cur_b in range(B):
            cur_cosine_sim = F.cosine_similarity(
                non_retained_tokens[cur_b][None, ...].unsqueeze(2), 
                retained_tokens[cur_b][None, ...].unsqueeze(1), dim=3)
            cosine_sim.append(cur_cosine_sim)
        cosine_sim = torch.cat(cosine_sim, dim=0)
        
        # cosine_sim = F.cosine_similarity(
        #     non_retained_tokens.unsqueeze(2), 
        #     retained_tokens.unsqueeze(1), dim=3)
        
        # import pdb;pdb.set_trace()
        
        # 480
        M = non_retained_tokens.shape[1]
        # 96
        N = retained_tokens.shape[1]
        
        wdist_sim = 1.0 - cosine_sim
        
        wdist = 1.0 * wdist_sim + 0.5 * wdist_spat
        # wdist = 1.0 * wdist_sim + 1.0 * wdist_spat
        
        xx = torch.zeros(
            1, M, dtype=cosine_sim.dtype, device=cosine_sim.device).fill_(1. / M)
        yy = torch.zeros(
            1, N, dtype=cosine_sim.dtype, device=cosine_sim.device).fill_(1. / N)

        eps = 0.1
        scaling = 1.0

        with torch.no_grad():
            KK = torch.exp(-wdist / eps)
            T = self.Sinkhorn_v1(KK, xx, yy, 100)
        
        
        
        agg = torch.einsum('bmn,bmd->bnd', T, non_retained_tokens)
        load = T.sum(dim=1, keepdim=False).unsqueeze(-1).clamp_min(1e-9)
        merged_retained_ota = (retained_tokens + scaling * agg) / (1.0 + scaling * load)
        
        
        nearest_token_indices = cosine_sim.argmax(dim=2)  # (B, N - T)
        # Track how many non-retained tokens merge with each retained token
        merge_count = torch.zeros(B, T_index[0], device=image_features.device, dtype=torch.int)
        # Merge tokens by averaging
        merged_features_vscan = torch.zeros_like(retained_tokens)  # (B, T, D)
        merged_features_vscan += retained_tokens * scaling
        # Process each non-retained token and add it to its nearest retained token
        expanded_indices = nearest_token_indices  # Shape: [B, N - T]
        merged_features_vscan.scatter_add_(1, nearest_token_indices.unsqueeze(-1).expand(-1, -1, D), non_retained_tokens)
        merge_count.scatter_add_(1, expanded_indices, torch.ones_like(expanded_indices, dtype=merge_count.dtype))
        # print(merge_count)

        # # Normalize the retained tokens by the number of non-retained tokens merging with them
        merged_features_vscan /= (scaling + merge_count.unsqueeze(2))
        # merged_features_vscan /= (merge_count.unsqueeze(2))
        
        merged_features = merged_features_vscan * 0.1 + merged_retained_ota * (1.0 - 0.1)
        # merged_features = merged_features_vscan * 0.2 + merged_retained_ota * (1.0 - 0.2)
        
        # import pdb;pdb.set_trace()
        
        for b in range(B):
            # Replace the non-retained tokens with the merged features
            image_features[b, index_mask[b]] = merged_features[b]
        
        return image_features

    
    def _robust_norm(self, C, eps=1e-6):
        # 批内两次中位数归一，避免某一项主导
        C = C.float()
        m1 = C.median(dim=1, keepdim=True).values
        m2 = m1.median(dim=2, keepdim=True).values if C.dim()==3 else m1
        return C / (m2 + eps)
    
    
    def ota_merge_N_to_M_time(self,
                        anchors,          # (B, M, D)  初始/当前 anchor tokens
                        sources,          # (B, N, D)  待聚合的源 tokens
                        anchors_ts,       # (B,M) 或 (M,)
                        sources_ts,       # (B,N) 或 (N,)
                        eps=0.1,          # Sinkhorn ε
                        max_iter=100,      # Sinkhorn 迭代
                        keep_ratio=0.5,       # 对行概率 p(n|m) 的阈值
                        sigma_t=1.5,      # 时间高斯σ（单位=帧）
                        max_dt=None,      # 超过该时间差直接屏蔽（None 关闭）
                        scaling=1.0,      # 融合步长系数
                        alpha=5.0,        # 语义项权重
                        gamma_t=0.5,      # 时间项权重
                        return_plan=False # 是否返回 Tplan
                        ):
        """
        将 sources (B,N,D) 通过 OTA 软权重聚合到 anchors (B,M,D)：
        - 计算 Tplan (B,N,M)，行归一得 p_row；
        - 对每个源 token：若 max_n p_row < thresh 则保留 (keep)；否则按整行 p_row 软分摊到所有 anchors；
        - 用 (anchor + scaling*agg) / (1 + scaling*load) 更新 anchors。
        返回:
        new_anchors: (B,M,D)
        keep_mask:   (B,N)  True=保留(低于阈值)，False=已参与聚合
        [可选] Tplan: (B,N,M)
        """
        device = anchors.device
        B, M, D = anchors.shape
        assert sources.shape[0] == B and sources.shape[2] == D, "batch 或维度不匹配"

        # === 1) 语义代价：1 - cos ===
        X = sources
        Y = anchors
        Xn = F.normalize(X, dim=-1)                              # (B,N,D)
        Yn = F.normalize(Y, dim=-1)                              # (B,M,D)
        cos = torch.bmm(Xn, Yn.transpose(1, 2)).clamp(-1+1e-6, 1-1e-6)  # (B,N,M)
        C_sim = 1.0 - cos                                            # (B,N,M)
        # # 数值稳：每个 batch 平移到最小为 0
        C_sim = C_sim - C_sim.amin(dim=(1,2), keepdim=True)
        
        # C = 1.0 - F.cosine_similarity(
        #     sources.unsqueeze(2), 
        #     anchors.unsqueeze(1), dim=3)
        
        dt = (sources_ts.unsqueeze(-1) - anchors_ts.unsqueeze(1)).abs()  # (B,N,M)
        C_time = 1.0 - torch.exp(-(dt*dt) / (2.0*(sigma_t**2) + 1e-9))   # ∈[0,1)
        if max_dt is not None:
            C_time = C_time + (dt > float(max_dt)).to(C_time.dtype) * 1e6

        C = C_sim * C_time
        
        
        # === 2) Sinkhorn 得运输计划 Tplan ===
        K = torch.exp(-C / eps)                                  # (B,N,M)
        mu = torch.full((B, X.shape[1]), 1.0 / X.shape[1], device=device, dtype=K.dtype)
        nu = torch.full((B, Y.shape[1]), 1.0 / Y.shape[1], device=device, dtype=K.dtype)
        with torch.no_grad():
            Tplan = self.Sinkhorn_v1(K, mu, nu, max_iter)        # (B,N,M)

        
        Tplan = Tplan.to(X.dtype)
        
        # === 3) 行归一 p(n|m) 并做阈值，决定保留/聚合 ===
        row_sum = Tplan.sum(dim=2, keepdim=True).clamp_min(1e-9) # (B,N,1)
        p_row   = Tplan / row_sum                                 # (B,N,M)
        pmax    = p_row.max(dim=2).values                         # (B,N)
        # keep_mask = (pmax < thresh)                               # True=保留

        tau_q = torch.quantile(pmax.double(), keep_ratio, dim=1, keepdim=True)
        keep_mask = pmax < tau_q
        
        # 仅让“需要聚合”的行参与（其余行置零）
        # p_row_sel = p_row * (~keep_mask).unsqueeze(-1).float()    # (B,N,M)
        p_row_sel = p_row * (~keep_mask).unsqueeze(-1)
        # import pdb;pdb.set_trace()
        # p_row_sel = p_row
         
        # === 4) 软权重聚合到 anchors，并用 scaling 融合 ===
        # agg[m] = sum_n p(n|m)*X[n];  load[m] = sum_n p(n|m)
        agg  = torch.bmm(p_row_sel.transpose(1, 2), X)            # (B,M,D)
        load = p_row_sel.sum(dim=1, keepdim=False).unsqueeze(-1)  # (B,M,1)

        # new_anchors = (Y + scaling * agg) / (1.0 + scaling * load.clamp_min(1e-9))
        new_anchors = (Y + scaling * agg) / (1.0 + scaling * load)

        if return_plan:
            return new_anchors, keep_mask, Tplan
        else:
            return new_anchors, keep_mask
    

    
    def encode_images_siglip_OTA(
        self, 
        images,
        fastvid_DySeg_c,
        fastvid_DySeg_tau,
        dynamic_seg,
        frame_num):
        # image_features, image_attentions = self.get_model().get_vision_tower()(images) # (B, N, C), (B, M, N) = (1, 729, 1152), (26, 1, 16, 729)

        if images.shape[0] <=32:
            image_features, image_attentions = self.get_model().get_vision_tower()(images) 
        else:
            image_features1, image_attentions1 = self.get_model().get_vision_tower()(images[0:32]) 
            image_features2, image_attentions2 = self.get_model().get_vision_tower()(images[32:]) 
            image_features = torch.cat((image_features1, image_features2), dim=0)
            image_attentions = torch.cat((image_attentions1, image_attentions2), dim=1)        
        
        # ######################FastVID#####################################################
        
        segment_sizes = None
        
        # # fastvid_DySeg_c = 8
        # # fastvid_DySeg_tau = 0.6
        # # frame_num = 32
        # segment_sizes = None
        # if dynamic_seg:
        #     frame_global_features, _ = self.get_model().get_vision_abstract()(image_features)
        #     frame_global_features = frame_global_features / frame_global_features.norm(dim=1, keepdim=True) 
        #     similarity_matrix = (frame_global_features[:-1] * frame_global_features[1:]).sum(dim=1)
        #     similarity_matrix = similarity_matrix.float()
        #     cut_indices_topk = torch.topk(similarity_matrix, fastvid_DySeg_c - 1, largest=False).indices
        #     cut_indices_cos = torch.nonzero(similarity_matrix < fastvid_DySeg_tau, as_tuple=False).squeeze(1)
        #     cut_indices = torch.unique(torch.cat([cut_indices_topk, cut_indices_cos])).sort().values
        #     padded = F.pad(cut_indices, (1, 1), value=-1)
        #     padded[-1] = frame_num - 1
        #     segment_sizes = padded.diff().tolist()
        #     rank0_print("segment_sizes : {}, input frames : {}".format(
        #         segment_sizes,
        #         frame_num))
        # # import pdb;pdb.set_trace()
        # #################################################################################
        
        image_attentions = image_attentions.mean(dim=2) # (26, 1, 729)
        B, N = image_features.shape[:2]

        visual_token_num = int(os.getenv("VISUAL_TOKEN_NUM", 126))
        
        # Retain all the tokens if visual_token_num is 576
        if visual_token_num == 729:
            index_mask = torch.ones(B, N, dtype=torch.bool, device=image_features.device)
            image_features = self.get_model().mm_projector(image_features) # (B, N, D)
            return image_features, index_mask, image_attentions[-2], None
        
        # Global and Local Scan
        image_attentions_shallow = image_attentions[5]
        image_attentions_deep = image_attentions[-2]

        # import pdb;pdb.set_trace()
        
        global_ratio = float(os.getenv("GLOBAL_RATIO", 0.5))
        local_ratio = 1 - global_ratio
        # Window [CLS] Attention
        # Shallow Layer: Local Tokens
        local_indices = self.window_cls_selection_siglip(
            image_attentions_shallow, 
            int(visual_token_num * local_ratio), 
            window_size=9)
        
        # # Deep Layer: Global Tokens
        for b in range(B):
            image_attentions_deep[b, local_indices[b]] = 0
        global_indices = torch.topk(
            image_attentions_deep, 
            k=visual_token_num -int(visual_token_num * local_ratio), dim=1)[1]

        token_indices = torch.cat((local_indices, global_indices), dim=1)
        
        # Generate index mask
        index_mask = torch.zeros(B, N, dtype=torch.bool, device=image_features.device) # (B, N)
        index_mask.scatter_(1, token_indices, True) # (B, N)

        # image_features = self.get_model().mm_projector(image_features) # (B, N, D)
        image_features = self.get_model().mm_projector(
            image_features.to(
                dtype=next(self.get_model().mm_projector.parameters()).dtype,
                device=next(self.get_model().mm_projector.parameters()).device
            )
        )
        # import pdb;pdb.set_trace()
        
        # Merge all other tokens into the selected tokens
        scaling = float(os.getenv("INTRA_SCALE", 1.0))
        image_features = self.token_merging_siglip_with_OTA(image_features, index_mask, scaling=scaling) # (B, N, D)
        
        image_attentions = image_attentions[-2]

        return image_features, index_mask, image_attentions, segment_sizes
    
    
    def encode_images(self, images):
        image_features, _ = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        # image_features = self.get_model().mm_projector(image_features)
        image_features = self.get_model().mm_projector(
            image_features.to(
                dtype=next(self.get_model().mm_projector.parameters()).dtype,
                device=next(self.get_model().mm_projector.parameters()).device
            )
        )

        return image_features
    
    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []
        all_faster_video_features = []
        cur_mm_spatial_pool_stride = self.config.mm_spatial_pool_stride

        for idx, feat in enumerate(per_videos_or_images_features):
            
            feat = self.get_model().mm_projector(feat)
            faster_video_feature = 0
            slower_img_feat = 0
            if idx in video_idx_in_batch and cur_mm_spatial_pool_stride > 1:
                slower_img_feat = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
                if self.config.add_faster_video:
                    cur_mm_spatial_pool_stride = cur_mm_spatial_pool_stride * 2
                    faster_video_feature = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
            if slower_img_feat is not 0:
                all_videos_or_images_features.append(slower_img_feat)
            else:
                all_videos_or_images_features.append(feat)
            all_faster_video_features.append(faster_video_feature)
        return all_videos_or_images_features,all_faster_video_features

    def add_token_per_grid(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        # image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = image_feature.flatten(2, 3)
        # image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        image_feature = torch.cat((
            image_feature, 
            self.model.image_newline[:, None, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        if getattr(self.config, "add_faster_video", False):
            # import pdb; pdb.set_trace()
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames,resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            # import pdb; pdb.set_trace()
            return image_feature
        # import pdb; pdb.set_trace()
        # image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        
        image_feature = image_feature.flatten(2, 3)
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature
    
    def add_token_per_grid_index(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        # image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = image_feature.flatten(2, 3)
        # image_feature = torch.cat((image_feature, 
        #                            torch.ones(1, dtype=torch.bool).to(image_feature.device).expand(*image_feature.shape[:-1], 1)), dim=-1)
        image_feature = torch.cat((
            image_feature, 
            torch.ones(1, dtype=torch.bool).to(image_feature.device).expand(*image_feature.shape[:-1], 1)), dim=-1)
        if getattr(self.config, "add_faster_video", False):
            # import pdb; pdb.set_trace()
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames,resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            # import pdb; pdb.set_trace()
            return image_feature
        # import pdb; pdb.set_trace()
        # image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        
        image_feature = image_feature.flatten(2, 3)
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        
        return image_feature

    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature

    def prepare_inputs_labels_for_multimodal(
        self, 
        input_ids, 
        position_ids, 
        attention_mask, 
        past_key_values, 
        labels, 
        images, 
        modalities=["image"], 
        texts=None,
        image_sizes=None):
        # import pdb;pdb.set_trace()
        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(modalities, str):
            modalities = [modalities]

        # import pdb; pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            
            input_frmaes = concat_images.shape[0]
            
            inter_compress = os.getenv('INTER_COMPRESS', "False") == 'True'
            dynamic_seg = os.getenv("DYNAMIC_SEGMENTS", "False") == "True"

            #####################OTA Sinknorn#####################
            (encoded_image_features,
             index_masks,
             image_attns,
             segment_clip_sizes) = self.encode_images_siglip_OTA(
                 concat_images,
                 dynamic_seg=dynamic_seg,
                 fastvid_DySeg_c=8,
                 fastvid_DySeg_tau=0.8,
                 frame_num=input_frmaes)
            # image_features,all_faster_video_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            
            # This is a list, each element is [num_images, patch * patch, dim]
            # rank_print(f"Concat images : {concat_images.shape}")
            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    image_features.append(self.get_2dPool(image_feat))
                else:
                    image_features.append(image_feat)
            index_masks = torch.split(index_masks, split_sizes)
            img_index_masks = []
            for idx, index_m in enumerate(index_masks):
                if idx in video_idx_in_batch:
                    img_index_masks.append(self.get_2dPool_index(index_m))
                else:
                    img_index_masks.append(image_feat)

            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

            def iter_segments(frames_num, dynamic_segment, segment_clip_sizes=None):
 
                if dynamic_segment:
                    assert segment_clip_sizes is not None
                    offset = 0
                    for seg_len in segment_clip_sizes:
                        yield offset, seg_len
                        offset += seg_len
                else:
                    if frames_num == 32:
                        interval = 4
                    elif frames_num == 64:
                        interval = 8
                    else:
                        raise ValueError("Invalid frame num, must be 32 or 64")
                    for anchor_idx in range(0, frames_num, interval):
                        yield anchor_idx, interval

            def temporal_ota_merge(
                image_feature,
                img_index_mask_fn,
                segments,
                keep_ratio,
                scaling,
            ):
                all_image_tokens = []

                for anchor_idx, seg_len in segments:
                    orig_anchors = (
                        image_feature[anchor_idx][img_index_mask_fn(anchor_idx)]
                        .unsqueeze(0)
                        .clone()
                    )
                    cur_anchors = orig_anchors.clone()
                    anchors_num = cur_anchors.shape[1]

                    # keep original anchor
                    all_image_tokens.append(orig_anchors[0])

                    if seg_len == 1:
                        continue

                    for inner_idx in range(1, seg_len):
                        cur_inner_img_feat = (
                            image_feature[anchor_idx + inner_idx][
                                img_index_mask_fn(anchor_idx + inner_idx)
                            ]
                            .unsqueeze(0)
                            .clone()
                        )

                        cur_inner_token_num = cur_inner_img_feat.shape[1]

                        anchors_ts = torch.zeros(
                            (1, anchors_num),
                            device=cur_anchors.device,
                            dtype=cur_anchors.dtype,
                        )
                        sources_ts = torch.ones(
                            (1, cur_inner_token_num),
                            device=cur_anchors.device,
                            dtype=cur_anchors.dtype,
                        ) * inner_idx

                        cur_anchors, keep_index_next = self.ota_merge_N_to_M_time(
                            cur_anchors,
                            cur_inner_img_feat,
                            anchors_ts=anchors_ts,
                            sources_ts=sources_ts,
                            eps=0.1,
                            keep_ratio=keep_ratio,
                            scaling=scaling,
                            max_iter=100,
                            max_dt=None,
                            sigma_t=1.5,
                        )

                        # residual tokens
                        all_image_tokens.append(
                            cur_inner_img_feat[0][keep_index_next[0]]
                        )

                    # merged anchor
                    all_image_tokens.append(cur_anchors[0])

                return torch.cat(all_image_tokens, dim=0)

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    # import pdb; pdb.set_trace()
                    # image_features : torch.Size([32, 196, 3584])
                    # video_idx_in_batch : [0]
                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        if mm_newline_position == "grid":
                            # llava video 7b here
                            # Grid-wise
                            image_feature = self.add_token_per_grid(image_feature)
                            img_index_mask = self.add_token_per_grid_index(
                                (img_index_masks[image_idx]).unsqueeze(dim=2)
                            )                          

                            if inter_compress:
                                segments = iter_segments(
                                    input_frmaes,
                                    dynamic_segment=dynamic_seg,
                                    segment_clip_sizes=segment_clip_sizes,
                                )

                                image_feature = temporal_ota_merge(
                                    image_feature=image_feature,
                                    img_index_mask_fn=lambda i: img_index_mask[i, :, 0],
                                    segments=segments,
                                    keep_ratio=float(os.environ.get("KEEP_RATIO", 0.4)),
                                    scaling=float(os.environ.get("INTRA_SCALE", 1.0)),
                                )
                            else:
                                all_image_tokens = []
                                for i in range(input_frmaes):
                                    token_per_frame = image_feature[i][img_index_mask[i].squeeze(-1).bool()]
                                    all_image_tokens.append(token_per_frame)
                                image_feature = torch.cat(all_image_tokens, dim=0)                          
                            
                            # import pdb; pdb.set_trace()
                            
                            if getattr(self.config, "add_faster_video", False): # False
                                faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                                # Add a token for each frame
                                concat_slow_fater_token = []
                                # import pdb; pdb.set_trace()
                                for _ in range(image_feature.shape[0]):
                                    if _ % self.config.faster_token_stride == 0:
                                        concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                    else:
                                        concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                # import pdb; pdb.set_trace()
                                image_feature = torch.cat(concat_slow_fater_token)

                            # # vid 13*13 + 13
                            # rank0_print("keep tokens : ", image_feature.shape[0], 
                            #       "compression ratio : {:.4f}".format(1.0 - (image_feature.shape[0]-13*input_frmaes) / (input_frmaes * 169)),
                            #       "Retenion ratio : {:.4f}".format((image_feature.shape[0]-13*input_frmaes) / (input_frmaes * 169)))
                        
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            image_feature = self.add_token_per_frame(image_feature)

                            new_image_features.append(image_feature.flatten(0, 1))
                            
                        elif mm_newline_position == "one_token":
                            
                            img_index_mask = img_index_masks[image_idx]

                            if inter_compress:
                                segments = iter_segments(
                                    input_frmaes,
                                    dynamic_segment=dynamic_seg,
                                    segment_clip_sizes=segment_clip_sizes,
                                )

                                image_feature = temporal_ota_merge(
                                    image_feature=image_feature,
                                    img_index_mask_fn=lambda i: img_index_mask[i, :],
                                    segments=segments,
                                    keep_ratio=float(os.getenv("KEEP_RATIO", 0.4)),
                                    scaling=float(os.getenv("INTER_SCALE", 1.0)),
                                )
                            else:
                                all_image_tokens = []
                                for i in range(input_frmaes):
                                    token_per_frame = image_feature[i][img_index_mask[i].squeeze(-1).bool()]
                                    all_image_tokens.append(token_per_frame)
                                image_feature = torch.cat(all_image_tokens, dim=0)  
                            # import pdb;pdb.set_trace()

                            if not os.getenv('EGOSCHEMA_UNPAD', "False")
                                if 'unpad' in mm_patch_merge_type:
                                    image_feature = torch.cat((
                                        image_feature,
                                        self.model.image_newline[None].to(image_feature.device)
                                    ), dim=0)
                            new_image_features.append(image_feature)

                            ###############################
                            
                            # # ov 14*14 + 1
                            # rank0_print("keep tokens : ", image_feature.shape[0], 
                            #       "compression ratio : {:.4f}".format(1.0 - (image_feature.shape[0] - 1) / (input_frmaes * 196)),
                            #       "Retenion ratio : {:.4f}".format((image_feature.shape[0] - 1) / (input_frmaes * 196)))
                            
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        # rank0_print("Single-images")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        new_image_features.append(image_feature)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                        new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        
        # import pdb; pdb.set_trace()
        
        rank0_print(f"final visual tokens number : {image_features[0].shape[0]}")
        
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

