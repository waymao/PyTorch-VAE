import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint 
from dataset import VAEDataset
from pytorch_lightning.plugins import DDPPlugin
import pdb
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/vae.yaml')
    parser.add_argument('-latent_dim', type=int, default = 128, help='latent dimension')
    parser.add_argument('-bt_lambda', type=float, default = 2.0, help='barlow twins lambda')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    config['model_params']['latent_dim'] = args.latent_dim
    config['model_params']['bt_lambda'] = args.bt_lambda
    print(config['model_params'])

    tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                name=config['model_params']['name'],)

    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)

    model = vae_models[config['model_params']['name']](**config['model_params'])
    experiment = VAEXperiment(model,
                            config['exp_params'])

    data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
    
    data.setup()

    runner = Trainer(logger=tb_logger,
                    callbacks=[
                        LearningRateMonitor(),
                        EarlyStopping(
                            monitor='val_loss', min_delta=0.00, patience=5, verbose=False, mode='min'
                        ),
                        ModelCheckpoint(save_top_k=2, 
                                        dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                        monitor= "val_loss",
                                        save_last= True)
                    ],
                    **config['trainer_params'])


    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)
    
    with open(f'{tb_logger.log_dir}/config.json', 'w') as fp:
        json.dump(config, fp)

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, datamodule=data)