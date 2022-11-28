import torch 
import numpy as np
import pandas as pd 
import yaml
import torchvision.utils as vutils
from models import *
import time

if __name__ == "__main__":
    f = r'./configs/vae.yaml'
    with open(f, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # print(config)
    model = vae_models[config['model_params']['name']](**config['model_params'])
    # model = model.load_from_checkpoint('logs/VanillaVAE/version_4/checkpoints/last.ckpt')
    model = torch.load('logs/VanillaVAE/version_4/checkpoints/last.ckpt')
    new_model = {}
    for key, val in model['state_dict'].items():
        new_model[key[6:]] = val
    # print(new_model)
    model = vae_models[config['model_params']['name']](**config['model_params'])
    model.load_state_dict(new_model)
    model = model.cuda()
    model.eval()
    
    t = time.time()
    sampled_images = model.sample(500, 'cuda')
    gt_images = 
    print('sample time', time.time() - t)
    # print(a)
    # print(a[0].shape)
    # vutils.save_image(a.data,
    #                 f"sample.png",
    #                 normalize=True,
    #                 nrow=10)