import torch
from torchvision.models import ShuffleNetV2
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights


class ShuffleFeatureExtractor(ShuffleNetV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        x = x.view(-1, self._stage_out_channels[-1])
        return x
    
if __name__ == "__main__":
    weights = ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
    pretrained_model = shufflenet_v2_x0_5(weights=weights)
    
    model = ShuffleFeatureExtractor(
        [len(pretrained_model.stage2),
         len(pretrained_model.stage3),
         len(pretrained_model.stage4)],
        pretrained_model._stage_out_channels)

    model.load_state_dict(pretrained_model.state_dict(), strict=False)
    
    # Create a sample input
    batch_size = 1
    channels = 3
    height = 224
    width = 224
    x = torch.randn(batch_size, channels, height, width)

    # Perform a forward pass
    with torch.no_grad():
        output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")