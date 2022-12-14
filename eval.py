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
from dataset import MyCelebA, VAEDataset, MyCIFAR10
from tqdm import tqdm
import json
import pdb

if __name__ == "__main__":
    # f = r'./configs/bt_vae.yaml'
    # with open(f, 'r') as file:
    #     try:
    #         config = yaml.safe_load(file)
    #     except yaml.YAMLError as exc:
    #         print(exc)
    
    ## define FID metric 
        

    num_image = 500
    
    ## load dataset
    transform_action = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(64),
                                            transforms.ToTensor(),])
    # data = MyCIFAR10('Data/cifar10', train='False', transform=transform_action, download=False)
    data = MyCelebA('Data/', split='test', transform=transform_action, download=False)
    first_num_image = torch.utils.data.Subset(data, range(num_image))
    gt_image = torch.stack([(x[0]*255).to(torch.uint8) for x in first_num_image])

    ## load model
    for version in range(5):
        model_dir = f'logs/CelebA/VanillaVAE/version_{version}'
        with open(os.path.join(model_dir, 'config.json'), 'r') as file:
            # load json config file 
            config = json.load(file)
        print('version', version, 'model_params', config['model_params'])
        model = vae_models[config['model_params']['name']](**config['model_params'])
        model = torch.load(os.path.join(model_dir, 'checkpoints/last.ckpt'))
        new_model = {}
        for key, val in model['state_dict'].items():
            new_model[key[6:]] = val
        model = vae_models[config['model_params']['name']](**config['model_params'])
        model.load_state_dict(new_model)
        model = model.cuda()
        model.eval()
        
        

        ## sample images 
        fid = FrechetInceptionDistance(feature=64)
        sampled_images = (model.sample(num_image, 'cuda') * 255).to(torch.uint8).cpu()

        
        # vutils.save_image(gt_image[:50], f"sample.png", normalize=True, nrow=10)
        fid.update(gt_image, real=True)
        fid.update(sampled_images, real=False)
        print('FID', fid.compute())
        # vutils.save_image(a.data,
        #                 f"sample.png",
        #                 normalize=True,
        #                 nrow=10)