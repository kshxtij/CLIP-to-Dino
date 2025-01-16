import torch
import torch.nn as nn
from transformers import CLIPModel, AutoProcessor, AutoImageProcessor, Dinov2ForImageClassification, AutoModel

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub.*')

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.CLIP = AutoModel.from_pretrained('facebook/dinov2-large')
        self.image_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
        self.mlp = nn.Sequential(nn.Linear(768, 768),
                                 nn.ReLU(),
                                 nn.Linear(768, 512))

        # Freeze CLIP
        for param in self.CLIP.parameters():
            param.requires_grad = False

    def preprocess_image(self, image):
        x = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        return x

    def forward(self, x):
        # x = self.CLIP.get_image_features(pixel_values=x)
        inputs = self.image_processor(images=x, return_tensors="pt")
        outputs = self.CLIP(**inputs)
        x = outputs.last_hidden_state
        x = self.mlp(x)
        return x