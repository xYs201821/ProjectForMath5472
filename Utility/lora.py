from diffusers import UNet2DConditionModel
from peft import LoraConfig, get_peft_model
import os
import torch
unet_lora_config = LoraConfig(
    r = 4,
    lora_alpha = 16,
    init_lora_weights = "olora",
    target_modules = ["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
)

class LoRa:
    def __init__(self, model_path, lora_rank=16, lora_alpha=32, dtype = torch.float16, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.model_path = os.path.expanduser(model_path)
        self._config_ = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.8,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
        )
        self.unet_phi = self.load_model(device, dtype)

    def load_model(self, device, dtype):
        unet_phi = UNet2DConditionModel.from_pretrained(
            os.path.join(self.model_path, "unet"), variant="fp16", torch_dtype=dtype
        ).to(device)
        unet_phi = get_peft_model(unet_phi, self._config_)
        return unet_phi
