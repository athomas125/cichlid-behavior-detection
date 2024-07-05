import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vit_h_14, ViT_H_14_Weights
from torchvision.models.vision_transformer import VisionTransformer
from PIL import Image

class ViTFeatureExtractor(VisionTransformer):            
    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # This is the encoded image. A normal vision transformer passes this through
        # a bunch of MLP heads however we just extract it for our model.
        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        return x[:, 0]



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
pretrained_model = vit_h_14(weights=weights)

model = ViTFeatureExtractor(
    image_size=pretrained_model.image_size,
    patch_size=pretrained_model.patch_size,
    num_layers=len(pretrained_model.encoder.layers),
    num_heads=pretrained_model.encoder.layers[0].num_heads,
    hidden_dim=pretrained_model.hidden_dim,
    mlp_dim=pretrained_model.mlp_dim
)

model.load_state_dict(pretrained_model.state_dict(), strict=False)

model = model.to(device)

image_path = "/data/home/athomas314/example_image.png"
vit_input = weights.transforms(Image.open(image_path))