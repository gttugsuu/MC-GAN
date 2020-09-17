------------ Options -------------
align_data: True
aspect_ratio: 1.0
base_font: False
base_root: ../datasets/Capitals64/BASE
batchSize: 1
blanks: 0.0
checkpoints_dir: ./checkpoints
conditional: True
conv3d: True
dataroot: ../datasets/public_web_fonts/ft37_1/
display_id: 0
display_winsize: 256
fineSize: 64
flat: False
gpu_ids: [0]
grps: 26
how_many: 1500
input_nc: 26
input_nc_1: 3
isTrain: False
loadSize: 64
max_dataset_size: inf
model: StackGAN
nThreads: 2
n_layers_D: 1
name: ft37_1_MCGAN_train
ndf: 64
ngf: 64
nif: 32
no_Style2Glyph: False
no_lsgan: False
no_permutation: False
norm: batch
ntest: inf
orna: False
output_nc: 26
output_nc_1: 3
partial: True
phase: test
print_weights: False
results_dir: ./results/
rgb: False
rgb_in: False
rgb_out: True
serial_batches: False
stack: False
use_dropout: False
use_dropout1: False
which_epoch: 400
which_epoch1: 700
which_model_netD: n_layers
which_model_netG: resnet_6blocks
which_model_preNet: 2_layers
-------------- End ----------------
------------ Options -------------
align_data: True
base_font: False
base_root: ../datasets/Capitals64/BASE
batchSize: 1
beta1: 0.5
blanks: 0.0
checkpoints_dir: ./checkpoints
conditional: True
continue_train: False
conv3d: True
dataroot: ../datasets/public_web_fonts/ft37_1/
display_freq: 100
display_id: 0
display_winsize: 256
fineSize: 64
flat: False
gamma: 0.0001
gpu_ids: [0]
grps: 26
input_nc: 26
input_nc_1: 3
isTrain: True
lambda_A: 100.0
lambda_C: 0.0
loadSize: 64
lr: 0.0002
max_dataset_size: inf
model: StackGAN
nThreads: 2
n_layers_D: 1
name: ft37_1_MCGAN_train
ndf: 64
nepoch: 200
ngf: 64
nif: 32
niter: 100
niter_decay: 100
no_Style2Glyph: False
no_html: False
no_lsgan: False
no_permutation: False
noisy_disc: False
norm: batch
orna: False
output_nc: 26
output_nc_1: 3
partial: True
phase: train
pool_size: 50
print_freq: 100
print_weights: False
rgb: False
rgb_in: False
rgb_out: True
save_epoch_freq: 5
save_latest_freq: 5000
serial_batches: False
stack: False
use_dropout: False
use_dropout1: False
which_epoch: 400
which_epoch1: 700
which_model_netD: n_layers
which_model_netG: resnet_6blocks
which_model_preNet: 2_layers
-------------- End ----------------
StackGAN
classname Conv3d
in random conv
classname BatchNorm3d
classname ReLU
classname Sequential
classname ResnetGenerator_3d_conv
resnet_6blocks
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname ReLU
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname ReLU
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname ReLU
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname ReLU
classname Dropout
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname Sequential
classname ResnetBlock
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname ReLU
classname Dropout
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname Sequential
classname ResnetBlock
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname ReLU
classname Dropout
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname Sequential
classname ResnetBlock
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname ReLU
classname Dropout
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname Sequential
classname ResnetBlock
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname ReLU
classname Dropout
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname Sequential
classname ResnetBlock
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname ReLU
classname Dropout
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname Sequential
classname ResnetBlock
classname ConvTranspose2d
in random conv
classname BatchNorm2d
in random batchnorm
classname ReLU
classname ConvTranspose2d
in random conv
classname BatchNorm2d
in random batchnorm
classname ReLU
classname Conv2d
in random conv
classname Tanh
classname Sequential
classname ResnetGenerator
resnet_6blocks
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname ReLU
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname ReLU
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname ReLU
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname ReLU
classname Dropout
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname Sequential
classname ResnetBlock
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname ReLU
classname Dropout
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname Sequential
classname ResnetBlock
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname ReLU
classname Dropout
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname Sequential
classname ResnetBlock
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname ReLU
classname Dropout
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname Sequential
classname ResnetBlock
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname ReLU
classname Dropout
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname Sequential
classname ResnetBlock
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname ReLU
classname Dropout
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname Sequential
classname ResnetBlock
classname Sequential
classname ResnetEncoder
resnet_6blocks
classname ConvTranspose2d
in random conv
classname BatchNorm2d
in random batchnorm
classname ReLU
classname ConvTranspose2d
in random conv
classname BatchNorm2d
in random batchnorm
classname ReLU
classname Conv2d
in random conv
classname Tanh
classname Sequential
classname ResnetDecoder
2 layers convolution applied before being fed into the discriminator
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname ReLU
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname ReLU
classname Sequential
classname InputTransformation
classname Conv2d
in random conv
classname LeakyReLU
classname Conv2d
in random conv
classname BatchNorm2d
in random batchnorm
classname LeakyReLU
classname Conv2d
in random conv
classname Sequential
classname NLayerDiscriminator
Load generators from their pretrained models...
model [StackGANModel] was created
process image... ../datasets/public_web_fonts/ft37_1/B/test/ft37_1.png
save to: ./results/ft37_1_MCGAN_train/test_400+700/images
