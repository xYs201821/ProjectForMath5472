from diffusers import UNet2DConditionModel
import torch
from torch import nn

class UNet:
    def __init__(
        self,
        sample_size=64,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        cross_attention_dim=768,
        attention_head_dim=8,
        dtype=torch.float16,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self._config_ = {
            "sample_size": sample_size,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "layers_per_block": layers_per_block,
            "block_out_channels": block_out_channels,
            "down_block_types": down_block_types,
            "up_block_types": up_block_types,
            "cross_attention_dim": cross_attention_dim,
            "attention_head_dim": attention_head_dim,
        }
        self.unet_phi = UNet2DConditionModel.from_config(self._config_)
        self.unet_phi = self.unet_phi.to(dtype)
        self.unet_phi = self.unet_phi.to(device)
