# %%
from geoclip.train import train
from geoclip.model import GeoCLIP
from datasets import load_dataset
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# %%
def img_train_transform():
    train_transform_list = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return train_transform_list

def img_val_transform():
    val_transform_list = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return val_transform_list

# %%
dataset = load_dataset('stochastic/random_streetview_images_pano_v0.0.2', split='train[:1%]')

# %%
# Convert the 'image' column to tensor
dataset = dataset.map(lambda x: {'image': transforms.PILToTensor()(x['image']).unsqueeze(0)}, num_proc=1)

# %%
# dataloader = GeoDataLoader('stochastic/random_streetview_images_pano_v0.0.2', split='train', transform=img_train_transform())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# %%
model = GeoCLIP()

# %%
optim = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-6)

# %%
train(dataloader, model, batch_size=64, device='cpu', optimizer=optim, epoch=1)

# %%
import pandas as pd

# %%
dataset.format

# %%
dataset[0]['image']

# %%



