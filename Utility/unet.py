from diffusers import UNet2DConditionModel
# Define the model configuration
    unet_config = {
        "sample_size": 64,  # The spatial size of the model's input and output
        "in_channels": 4,   # Number of input channels (3 for RGB + 1 for time embedding)
        "out_channels": 4,  # Number of output channels
        "layers_per_block": 2,  # Number of resnet blocks per downsample/upsample
        "block_out_channels": (64, 128, 256, 512),  # Number of output channels for each UNet block
        "down_block_types": (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        "up_block_types": (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        "cross_attention_dim": 512,  # Dimension of the cross-attention feature
        "attention_head_dim": 8,     # Number of attention heads
    }

# Create the UNet2DConditionModel
    unet_phi = UNet2DConditionModel(**unet_config)
    # Load the pretrained model
    pretrained_model_path = "/path/to/pretrained/model"

    # Initialize the new UNet with the pretrained weights
    unet_phi.load_state_dict(torch.load(pretrained_model_path))