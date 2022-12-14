import torch 
import numpy as np
import pandas as pd 
import yaml
import torchvision.utils as vutils
from torchvision.io import read_image
from models import *
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from torchvision import transforms
from PIL.Image import open as open_image
import time
from dataset import MyCelebA, VAEDataset, MyCIFAR10
from tqdm import tqdm
import json
import pdb
from matplotlib import pyplot as plt

# with open(os.path.join(root, "list_attr_celeba.txt"), "r") as f:
#     list_attr_celeba = f.read().splitlines()
#     attribute_descriptions = list_attr_celeba[1]
#     list_attr_celeba = list_attr_celeba[2:]

#     list_attr_celeba = np.array(
#         [[int(x) for x in s[11:].split()] for s in list_attr_celeba]
#     )

def load_model(model_dir: str):
    '''
    given a model directory, return the model

    Parameters
        model_dir: directory of the model

    Returns
        model
    '''
    with open(os.path.join(model_dir, 'config.json'), 'r') as file:
        # load json config file 
        config = json.load(file)
    model = vae_models[config['model_params']['name']](**config['model_params'])
    model = torch.load(os.path.join(model_dir, 'checkpoints/last.ckpt'))
    new_model = {}
    for key, val in model['state_dict'].items():
        new_model[key[6:]] = val
    model = vae_models[config['model_params']['name']](**config['model_params'])
    model.load_state_dict(new_model)
    # model = model.cuda()
    model.eval()
    print('model loaded')
    
    return model 

def load_attr(attr_dir: str):
    '''
    given the attribute directory, return the attribute file

    Parameters
        attr_dir: directory of the attribute file

    Returns
        attribute file
    '''

    root = 'Data/celeba'
    df = pd.read_csv(os.path.join(root, 'list_attr_celeba.txt'), delim_whitespace=True, header=1)
    print('attribute file loaded')

    return df

def show_latent(file_names: list, model):
    '''
    given a list of indices, return the latent vectors of the 
    corresponding images via the model

    Parameters
        file_names: list of file names
        model: vae model to use

    Returns
        list of latent vectors of the images
    '''
    image_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(64),
                                            transforms.ToTensor(),])
    img_inputs = []
    for file in file_names:
        img = open_image(os.path.join('Data/celeba/img_align_celeba', file))
        img_inputs.append(image_transform(img))
    x = torch.stack(img_inputs)
    print(x.shape)
    latent_vars = model.encode(x)
    return latent_vars[0]


def visualize_latent(latent: list, attr: pd.DataFrame):
    '''
    given a list of latent vectors and the attribute file, 
    visualize the latent vectors with respect to attributes

    Parameters
        latent: list of latent vectors
        attr: attribute file

    Returns
        None
    '''
    # generate dummy data
    latent = np.random.randn(200, 128)
    print(latent)

    pass

if __name__ == '__main__':
    # read attribute file
    attr_dir = 'Data/celeba'
    attr = load_attr(attr_dir)

    # read model 
    version = 0
    model_dir = f'logs/CelebA/VanillaVAE/version_{version}'
    model = load_model(model_dir)

    with open(os.path.join(model_dir, 'config.json'), 'r') as file:
        # load json config file 
        config = json.load(file)

    data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
    data.setup()
    pdb.set_trace()

    # sample images
    sampled = attr.sample(200, random_state=152)
    file_name = list(sampled.index)

    # get latent vectors
    latent = show_latent(file_name, model)

    # visualie latent vectors with respect to attributes
    visualize_latent(latent, attr)