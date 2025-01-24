import torch
import torch.nn as nn
from transformers import CLIPModel, AutoProcessor, CLIPTokenizer

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub.*')

class ImageEncoder(nn.Module):
    def __init__(self, query_images=True):
        super(ImageEncoder, self).__init__()
        self.CLIP = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.text_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = nn.Sequential(nn.Linear(768, 768),
                                 nn.ReLU(),
                                 nn.Linear(768, 512))
        
        self.query_images = query_images

        # Freeze CLIP
        for param in self.CLIP.parameters():
            param.requires_grad = False

    def to(self, device):
        self.CLIP.to(device)
        self.mlp.to(device)
        return super().to(device)

    def preprocess_image(self, image):
        x = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        return x
    
    def preprocess_text(self, text):
        x = self.text_tokenizer(text, return_tensors="pt")["input_ids"]
        return x

    def forward(self, x):
        if self.query_images:
            x = self.CLIP.get_image_features(pixel_values=x)
        else:
            x = self.CLIP.get_text_features(input_ids=x)
        x = self.mlp(x)
        return x