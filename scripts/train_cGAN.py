import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-DATA', type=str, default='Capitals64')
args = parser.parse_args()

#=====================================
## Set Parameters
#=====================================

DATA = args.DATA
DATASET = f"../datasets/{DATA}/"
experiment_dir = "GlyphNet_pretrain"
MODEL="cGAN"
MODEL_G="resnet_6blocks"
MODEL_D="n_layers"
n_layers_D=1
NORM="batch"
IN_NC=26
O_NC=26
GRP=26
PRENET="2_layers"
FINESIZE=64
LOADSIZE=64
LAM_A=100
NITER=500
NITERD=100
BATCHSIZE=150
CUDA_ID=0

if not os.path.isdir(f"./checkpoints/{experiment_dir}"):
    os.mkdir(f"./checkpoints/{experiment_dir}")

LOG = f"./checkpoints/{experiment_dir}/output.txt"

if os.path.isfile(LOG):
    os.remove(LOG)

command = f"python train.py --dataroot {DATASET} --name {experiment_dir} --model {MODEL} --which_model_netG {MODEL_G} --which_model_netD {MODEL_D}  --n_layers_D {n_layers_D} --which_model_preNet {PRENET} --norm {NORM} --input_nc {IN_NC} --output_nc {O_NC} --grps {GRP} --fineSize {FINESIZE} --loadSize {LOADSIZE} --lambda_A {LAM_A} --align_data --use_dropout --display_id 0 --niter {NITER} --niter_decay {NITERD} --batchSize {BATCHSIZE} --conditional --save_epoch_freq 100 --print_freq 100 --conv3d"

os.system(command)
