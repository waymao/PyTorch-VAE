#! /bin/sh
# python3 run.py -c configs/vae.yaml -latent_dim 2 
# python3 run.py -c configs/vae.yaml -latent_dim 16 
# python3 run.py -c configs/vae.yaml -latent_dim 32 
# python3 run.py -c configs/vae.yaml -latent_dim 64 
# python3 run.py -c configs/vae.yaml -latent_dim 128 

for bt_lambda in 0.00025 0.0025 0.025 0.25 -0.00025;
    do for latent_dim in 2 16 32 64 128;
        do python3 run.py -c configs/bt_vae.yaml -latent_dim $latent_dim -bt_lambda $bt_lambda;
    done;
done