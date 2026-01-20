# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import re

import torch
import torch.nn as nn
from timm.models.layers import Mlp
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel

MOL_FP_PAD_VALUE = -1


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def pad_to_multiple(x, multiple):
    n, w, h, c = x.size()
    pad_w = (multiple - (w % multiple)) % multiple
    pad_h = (multiple - (h % multiple)) % multiple
    if pad_w:
        x = torch.cat([x, x.new_zeros((n, pad_w, h, c))], dim=1).contiguous()
        w += pad_w
    if pad_h:
        x = torch.cat([x, x.new_zeros((n, w, pad_h, c))], dim=2).contiguous()
    return x


def merge_square_blocks(x, block_size):
    """Downsample a (N, W, H, C) grid by merging block_size x block_size patches."""
    x = pad_to_multiple(x, block_size)
    n, w, h, c = x.size()
    x = x.view(n, w, h // block_size, c * block_size)
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(n, h // block_size, w // block_size, c * block_size * block_size)
    x = x.permute(0, 2, 1, 3).contiguous()
    return x


class DownSampleBlock(nn.Module):
    def forward(self, x):
        vit_embeds = x
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = merge_square_blocks(vit_embeds, block_size=2)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return vit_embeds


class DownSample2x2BlockFix(nn.Module):
    def forward(self, x):
        vit_embeds = x
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = merge_square_blocks(vit_embeds, block_size=2)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return vit_embeds


class DownSample3x3BlockFix(nn.Module):
    def forward(self, x):
        vit_embeds = x
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = merge_square_blocks(vit_embeds, block_size=3)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return vit_embeds


class MultimodalProjectorConfig(PretrainedConfig):
    model_type = "v2l_projector"

    def __init__(self, mm_projector_type: str = None, **kwargs):
        super().__init__()
        self.mm_projector_type = mm_projector_type


class MultimodalProjector(PreTrainedModel):
    config_class = MultimodalProjectorConfig

    def __init__(self, mm_projector_cfg: MultimodalProjectorConfig, config: PretrainedConfig):
        super().__init__(mm_projector_cfg)
        mm_projector_type = mm_projector_cfg.mm_projector_type
        self.mm_projector_type = mm_projector_type
        self.downsample_rate = 1
        
        n_embed = config.hidden_size
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        torch.manual_seed(42)
        self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
        self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)

        if mm_projector_type == "identity":
            self.layers = IdentityMap()
        elif mm_projector_type == "linear":
            self.layers = nn.Linear(2048, config.hidden_size)
        elif mm_projector_type == "fusion":
            hidden_size = config.hidden_size
            
            # Get input dimensions from config (avoid hardcoding)
            image_input_dim = getattr(config, "mm_hidden_size", 2048)
            # E3FP fingerprint bits (size of dense fingerprint vector)
            mol_fp_bits = int(getattr(config, "mol_fp_bits", 4096))
            
            # Store fingerprint size for use in forward pass
            self.mol_fp_bits = mol_fp_bits
            
            # Attention heads configuration
            num_heads = getattr(config, "mm_projector_num_heads", None) or getattr(config, "num_attention_heads", None)
            if num_heads is None:
                num_heads = 8
            if hidden_size % num_heads != 0:
                for candidate in (8, 4, 2, 1):
                    if hidden_size % candidate == 0:
                        num_heads = candidate
                        break
            
            # Dropout and normalization config
            dropout_rate = getattr(config, "mm_projector_dropout", 0.0)
            eps = getattr(config, "layer_norm_eps", 1e-5)
            
            # Build projection layers
            self.image_proj = nn.Linear(image_input_dim, hidden_size)
            
            # E3FP hash embedding: indices in [0, mol_fp_bits - 1], padding at -1 mapped to 0.
            self.molecule_embed = nn.Embedding(mol_fp_bits + 1, hidden_size, padding_idx=0)

            self.initialize_molecule_embedding(self.molecule_embed, std=hidden_size**-0.5)

            # Cross-attention with dropout
            self.cross_attn = nn.MultiheadAttention(
                hidden_size, num_heads, 
                dropout=dropout_rate,
                batch_first=True
            )
            self.cross_attn_norm = nn.LayerNorm(hidden_size, eps=eps)
            
            # FFN with dropout
            ffn_layers = [
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
            ]
            if dropout_rate > 0:
                ffn_layers.append(nn.Dropout(dropout_rate))
            ffn_layers.extend([
                nn.Linear(hidden_size * 4, hidden_size),
            ])
            if dropout_rate > 0:
                ffn_layers.append(nn.Dropout(dropout_rate))
            self.ffn = nn.Sequential(*ffn_layers)
            self.ffn_norm = nn.LayerNorm(hidden_size, eps=eps)
            
            # Initialize weights properly
            # Image projection: Xavier uniform initialization
            nn.init.xavier_uniform_(self.image_proj.weight)
            if self.image_proj.bias is not None:
                nn.init.zeros_(self.image_proj.bias)
            
            # FFN layers: Kaiming initialization for better gradient flow with GELU
            for module in self.ffn.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        elif mm_projector_type == "concat":
            # Get input dimensions from config (avoid hardcoding)
            image_input_dim = getattr(config, "mm_hidden_size", 2048)
            
            # E3FP fingerprint configuration
            mol_fp_bits = int(getattr(config, "mol_fp_bits", 4096))
            self.mol_fp_bits = mol_fp_bits
            
            # Molecular fingerprint embedding dimension (can be smaller than hidden_size)
            mol_embed_dim = getattr(config, "mol_embed_dim", config.hidden_size // 4)
            
            # E3FP hash embedding shared with the fusion projector.
            self.molecule_embed = nn.Embedding(mol_fp_bits + 1, mol_embed_dim, padding_idx=0)

            self.initialize_molecule_embedding(self.molecule_embed, std=mol_embed_dim**-0.5)

            # Projection layers: concatenate image features with pooled molecular embeddings
            self.layers = nn.Sequential(
                nn.Linear(image_input_dim + mol_embed_dim, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
        elif mm_projector_type == "mlp_downsample":
            self.layers = nn.Sequential(
                DownSampleBlock(),
                nn.LayerNorm(config.mm_hidden_size * 4),
                nn.Linear(config.mm_hidden_size * 4, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
            self.downsample_rate = 2
        elif mm_projector_type == "mlp_downsample_2x2_fix":
            self.layers = nn.Sequential(
                DownSample2x2BlockFix(),
                nn.LayerNorm(config.mm_hidden_size * 4),
                nn.Linear(config.mm_hidden_size * 4, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
            self.downsample_rate = 2
        elif mm_projector_type == "mlp_downsample_3x3_fix":
            self.layers = nn.Sequential(
                DownSample3x3BlockFix(),
                nn.LayerNorm(config.mm_hidden_size * 9),
                nn.Linear(config.mm_hidden_size * 9, config.mm_hidden_size * 3),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size * 3),
                nn.Linear(config.mm_hidden_size * 3, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
            self.downsample_rate = 3
        elif mm_projector_type == "mlp_downsample_3x3_s2":
            self.layers = nn.Sequential(
                DownSample3x3BlockFix(),
                nn.LayerNorm(config.mm_hidden_size * 9),
                nn.Linear(config.mm_hidden_size * 9, config.mm_hidden_size * 3),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size * 3),
                nn.Linear(config.mm_hidden_size * 3, config.mm_hidden_size),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size),
                nn.Linear(config.mm_hidden_size, config.mm_hidden_size // 3),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size // 3),
                nn.Linear(config.mm_hidden_size // 3, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
        elif mm_projector_type == "mlp_downsample_3x3_s2_new":
            self.layers = nn.Sequential(
                DownSample3x3BlockFix(),
                nn.LayerNorm(config.mm_hidden_size * 9),
                nn.Linear(config.mm_hidden_size * 9, config.mm_hidden_size * 4),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size * 4),
                nn.Linear(config.mm_hidden_size * 4, config.mm_hidden_size * 2),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size * 2),
                nn.Linear(config.mm_hidden_size * 2, config.mm_hidden_size),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size),
                nn.Linear(config.mm_hidden_size, config.mm_hidden_size // 3),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size // 3),
                nn.Linear(config.mm_hidden_size // 3, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
        else:
            mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", mm_projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU())
                    modules.append(nn.Linear(config.hidden_size, config.hidden_size))
                self.layers = nn.Sequential(*modules)
            else:
                raise ValueError(f"Unknown projector type: {mm_projector_type}")

        if getattr(config, "ps3", False):
            if getattr(config, "look_close_mode", None) == "after_prompt":
                if getattr(config, "top_down_prompt_head_type", "linear") == "linear":
                    self.top_down_prompt_head = nn.Linear(config.hidden_size, config.mm_hidden_size)
                elif getattr(config, "top_down_prompt_head_type", "linear") == "mlp":
                    self.top_down_prompt_head = Mlp(
                        in_features=config.hidden_size,
                        hidden_features=config.mm_hidden_size * 2,
                        out_features=config.mm_hidden_size,
                        norm_layer=nn.LayerNorm,
                    )
                else:
                    raise NotImplementedError

                for n, p in self.top_down_prompt_head.named_parameters():
                    if "norm" not in n:
                        p.data.uniform_(-0.02, 0.02)

            if getattr(config, "high_res_pos_embed", False):
                self.high_res_pos_embed = nn.Parameter(torch.zeros(1, config.mm_low_res_token_num, config.hidden_size))
                self.high_res_scale_embed = nn.ParameterList(
                    [nn.Parameter(torch.zeros(1, 1, config.hidden_size)) for _ in range(config.mm_scale_num)]
                )

    def initialize_molecule_embedding(self, embedding, std):
        """Initialize embedding weights with a zeroed padding row."""
        def init_weights(tensor):
            nn.init.normal_(tensor, mean=0.0, std=std)
            tensor[0].fill_(0.0)

        with torch.no_grad():
            if embedding.weight.numel() == 0:
                try:
                    import deepspeed
                    with deepspeed.zero.GatheredParameters([embedding.weight], modifier_rank=0):
                        if embedding.weight.numel() > 0:
                            init_weights(embedding.weight)
                except (ImportError, AttributeError):
                    pass
            else:
                init_weights(embedding.weight)

    def normalize_molecule_fp(self, molecule_fp, batch_size, device, context):
        if molecule_fp is None:
            raise ValueError(f"{context} requires molecule_fp.")

        mol_fp = molecule_fp if isinstance(molecule_fp, torch.Tensor) else torch.as_tensor(molecule_fp)
        if mol_fp.dim() == 1:
            mol_fp = mol_fp.view(1, 1, -1)
        elif mol_fp.dim() == 2:
            if mol_fp.shape[0] == batch_size:
                mol_fp = mol_fp.unsqueeze(1)
            else:
                mol_fp = mol_fp.unsqueeze(0)
        elif mol_fp.dim() != 3:
            raise ValueError(f"E3FP fingerprint must be 1D, 2D, or 3D tensor, got shape {mol_fp.shape}")

        if mol_fp.shape[0] != batch_size:
            if mol_fp.shape[0] == 1:
                mol_fp = mol_fp.expand(batch_size, -1, -1)
            else:
                raise ValueError(
                    f"molecule_fp batch size {mol_fp.shape[0]} must match image features {batch_size}."
                )

        return mol_fp.to(device=device)

    def build_molecule_indices(self, mol_fp):
        if mol_fp.dtype.is_floating_point:
            non_integer = (mol_fp != mol_fp.floor()) & (mol_fp != MOL_FP_PAD_VALUE)
            if non_integer.any():
                raise ValueError(
                    f"E3FP fingerprint values must be integers in [0, {self.mol_fp_bits - 1}] "
                    f"or {MOL_FP_PAD_VALUE} for padding."
                )

        mol_indices = mol_fp.long()
        padding_mask = mol_indices == MOL_FP_PAD_VALUE

        valid_hashes = mol_indices[~padding_mask]
        if valid_hashes.numel() > 0:
            min_hash = valid_hashes.min().item()
            max_hash = valid_hashes.max().item()
            if min_hash < 0:
                raise ValueError(
                    f"E3FP hash indices must be >= 0 or {MOL_FP_PAD_VALUE} for padding. "
                    f"Found invalid value: {min_hash}."
                )
            if max_hash >= self.mol_fp_bits:
                raise ValueError(
                    f"E3FP hash indices must be < mol_fp_bits={self.mol_fp_bits}, got {max_hash}. "
                    "Ensure mol_fp_bits matches the E3FP generation parameter."
                )

        mol_indices = mol_indices.clone()
        mol_indices[~padding_mask] = mol_indices[~padding_mask] + 1
        mol_indices[padding_mask] = 0
        return mol_indices, padding_mask

    def forward_fusion(self, x, molecule_fp):
        """Fuse image features with molecular fingerprints via cross-attention."""
        if not isinstance(x, torch.Tensor):
            raise ValueError("Fusion projector expects image features as a Tensor.")

        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True
        elif x.dim() != 3:
            raise ValueError(f"Fusion projector expects (B, L, C) features, got {x.shape}.")

        batch_size = x.shape[0]
        mol_fp = self.normalize_molecule_fp(molecule_fp, batch_size, x.device, "Fusion projector")
        mol_indices, padding_mask = self.build_molecule_indices(mol_fp)

        mol_embeds = self.molecule_embed(mol_indices)
        batch_size, num_atoms, num_levels, hidden = mol_embeds.shape
        mol_features = mol_embeds.reshape(batch_size, num_atoms * num_levels, hidden)

        pad_mask = padding_mask.reshape(batch_size, num_atoms * num_levels)
        all_pad_mask = pad_mask.all(dim=1)
        if all_pad_mask.any():
            # Avoid NaNs when all tokens are masked.
            pad_mask = pad_mask.clone()
            pad_mask[all_pad_mask, 0] = False

        image_features = self.image_proj(x)
        fused_features, _ = self.cross_attn(
            query=image_features,
            key=mol_features,
            value=mol_features,
            key_padding_mask=pad_mask,
        )
        image_features = self.cross_attn_norm(image_features + fused_features)
        image_features = self.ffn_norm(image_features + self.ffn(image_features))

        return image_features.squeeze(0) if squeeze_output else image_features

    def forward_concat(self, x, molecule_fp):
        """Pool molecular embeddings and concatenate with image features."""
        if not isinstance(x, torch.Tensor):
            raise ValueError("Concat projector expects image features as a Tensor.")

        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True
        elif x.dim() != 3:
            raise ValueError(f"Concat projector expects (B, L, C) features, got {x.shape}.")

        batch_size = x.shape[0]
        mol_fp = self.normalize_molecule_fp(molecule_fp, batch_size, x.device, "Concat projector")
        mol_indices, padding_mask = self.build_molecule_indices(mol_fp)

        mol_embeds = self.molecule_embed(mol_indices)
        batch_size, num_atoms, num_levels, mol_embed_dim = mol_embeds.shape
        flat_embeds = mol_embeds.reshape(batch_size, num_atoms * num_levels, mol_embed_dim)
        flat_padding_mask = padding_mask.reshape(batch_size, num_atoms * num_levels)

        valid_count = (~flat_padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
        masked_embeds = flat_embeds * (~flat_padding_mask).unsqueeze(-1).float()
        mol_vec = masked_embeds.sum(dim=1) / valid_count

        mol_vec_expanded = mol_vec.unsqueeze(1).expand(-1, x.shape[1], -1)
        concat_features = torch.cat([x, mol_vec_expanded], dim=-1)
        out = self.layers(concat_features)

        return out.squeeze(0) if squeeze_output else out

    def forward(self, x, forward_top_down_prompt_head=False, molecule_fp=None, *args, **kwargs):
        if forward_top_down_prompt_head:
            return self.top_down_prompt_head(x)

        if isinstance(x, torch.Tensor):
            if self.mm_projector_type == "fusion":
                return self.forward_fusion(x, molecule_fp)
            if self.mm_projector_type == "concat":
                return self.forward_concat(x, molecule_fp)
            return self.layers(x)

        uses_molecule = self.mm_projector_type in ("fusion", "concat")
        if uses_molecule and molecule_fp is None:
            raise ValueError(f"{self.mm_projector_type} projector requires molecule_fp for list input.")
        if uses_molecule and molecule_fp is not None and len(x) != len(molecule_fp):
            raise ValueError(f"Batch size mismatch: got {len(x)} images but {len(molecule_fp)} molecule_fps")

        def project_features(features, mol_fp):
            if self.mm_projector_type == "fusion":
                return self.forward_fusion(features, mol_fp)
            if self.mm_projector_type == "concat":
                return self.forward_concat(features, mol_fp)
            return self.layers(features)

        images_in_this_batch = []
        for i, item in enumerate(x):
            if isinstance(item, list) and item:
                item = item[0]

            mol_fp_i = molecule_fp[i:i + 1] if uses_molecule else None
            if item["local_features"] is None:
                global_features = project_features(item["global_features"], mol_fp_i)

                _, hw, n_dim = global_features.shape
                h = w = int(hw ** 0.5)

                global_features = global_features.view(h, w, n_dim)
                global_features = torch.cat(
                    [global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1
                )
                global_features = global_features.view(-1, n_dim)

                global_local_features = torch.cat([global_features, self.view_seperator[None, :]], dim=0)
            else:
                global_features = item["global_features"]
                local_features = item["local_features"]
                crop_shape = item["crop_shape"]

                global_features = project_features(global_features, mol_fp_i)
                local_features = project_features(local_features, mol_fp_i)

                # Reshape Global
                _, hw, n_dim = global_features.shape
                h = w = int(hw ** 0.5)
                global_features = global_features.view(h, w, n_dim)
                global_features = torch.cat(
                    [global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1
                )
                global_features = global_features.view(-1, n_dim)

                # Reshape Local
                _2, hw2, n_dim2 = local_features.shape
                h2 = w2 = int(hw2 ** 0.5)
                width_crop_num, height_crop_num = crop_shape[0], crop_shape[1]

                local_features = (
                    local_features.view(height_crop_num, width_crop_num, h2, w2, n_dim2)
                    .permute(0, 2, 1, 3, 4)
                    .reshape(height_crop_num * h2, width_crop_num * w2, n_dim2)
                )
                local_features = torch.cat(
                    [local_features, self.image_newline[None, None, :].expand(height_crop_num * h2, 1, n_dim2)], dim=1
                )
                local_features = local_features.view(-1, n_dim2)

                global_local_features = torch.cat([local_features, global_features, self.view_seperator[None, :]], dim=0)

            images_in_this_batch.append(global_local_features)

        return images_in_this_batch

AutoConfig.register("v2l_projector", MultimodalProjectorConfig)
AutoModel.register(MultimodalProjectorConfig, MultimodalProjector)
