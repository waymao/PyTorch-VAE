import torch 
import numpy as np
import pandas as pd 
import yaml
import torchvision.utils as vutils
from models import *
from torchmetrics.image.fid import FrechetInceptionDistance
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from torchvision import transforms
import time
from dataset import MyCelebA, VAEDataset
from tqdm import tqdm

if __name__ == "__main__":
    f = r'./configs/vae.yaml'
    with open(f, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    ## load model
    model = vae_models[config['model_params']['name']](**config['model_params'])
    model = torch.load('logs/VanillaVAE/version_4/checkpoints/last.ckpt')
    new_model = {}
    for key, val in model['state_dict'].items():
        new_model[key[6:]] = val
    model = vae_models[config['model_params']['name']](**config['model_params'])
    model.load_state_dict(new_model)
    model = model.cuda()
    model.eval()
    
    ## define FID metric 
    fid = FrechetInceptionDistance(feature=64)

    ## sample images 
    num_image = 500
    sampled_images = (model.sample(num_image, 'cuda') * 255).to(torch.uint8).cpu()

    ## load dataset
    transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(64),
                                            transforms.ToTensor(),])
    data = MyCelebA('Data/', split='test', transform=transforms, download=False)
    for idx, image in tqdm(enumerate(iter(data))):
        fid.update((image[0][None,:]*255).to(torch.uint8), real=True)

        if idx == num_image:
            break
    # data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
    # dataloader = data.test_dataloader()
    # for step, x in enumerate(dataloader):
        # fid.update(x, real=True)

    fid.update(sampled_images, real=False)
    print('FID', fid.compute())

    # vutils.save_image(a.data,
    #                 f"sample.png",
    #                 normalize=True,
    #                 nrow=10)