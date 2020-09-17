TRAIN MODEL WITH REAL TRAINING DATA
------------ Options -------------
align_data: True
base_font: True
base_root: ../datasets/Capitals64/BASE
batchSize: 7
beta1: 0.5
blanks: 0.0
checkpoints_dir: ./checkpoints
conditional: True
continue_train: False
conv3d: True
dataroot: ../datasets/public_web_fonts/ft37_1/
display_freq: 5
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
lambda_A: 300.0
lambda_C: 10.0
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
niter: 400
niter_decay: 300
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
print_freq: 5
print_weights: False
rgb: False
rgb_in: False
rgb_out: True
save_epoch_freq: 100
save_latest_freq: 5000
serial_batches: False
stack: False
use_dropout: True
use_dropout1: False
which_epoch: 400
which_epoch1: 0
which_model_netD: n_layers
which_model_netG: resnet_6blocks
which_model_preNet: 2_layers
-------------- End ----------------
------------ Options -------------
align_data: True
base_font: True
base_root: ../datasets/Capitals64/BASE
batchSize: 7
beta1: 0.5
blanks: 0.0
checkpoints_dir: ./checkpoints
conditional: True
continue_train: False
conv3d: True
dataroot: ../datasets/public_web_fonts/ft37_1/
display_freq: 5
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
lambda_A: 300.0
lambda_C: 10.0
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
niter: 400
niter_decay: 300
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
print_freq: 5
print_weights: False
rgb: False
rgb_in: False
rgb_out: True
save_epoch_freq: 100
save_latest_freq: 5000
serial_batches: False
stack: False
use_dropout: True
use_dropout1: False
which_epoch: 400
which_epoch1: 0
which_model_netD: n_layers
which_model_netG: resnet_6blocks
which_model_preNet: 2_layers
-------------- End ----------------
#training images = 5
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
---------- Networks initialized -------------
ResnetGenerator_3d_conv(
  (model): Sequential(
    (0): Conv3d(26, 26, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=26)
    (1): BatchNorm3d(26, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
)
Total number of parameters: 780
ResnetGenerator(
  (model): Sequential(
    (0): Conv2d(26, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(64, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (4): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): Conv2d(192, 576, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (7): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU(inplace=True)
    (9): ResnetBlock(
      (conv_block): Sequential(
        (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.0, inplace=False)
        (4): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (10): ResnetBlock(
      (conv_block): Sequential(
        (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.0, inplace=False)
        (4): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (11): ResnetBlock(
      (conv_block): Sequential(
        (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.0, inplace=False)
        (4): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (12): ResnetBlock(
      (conv_block): Sequential(
        (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.0, inplace=False)
        (4): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (13): ResnetBlock(
      (conv_block): Sequential(
        (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.0, inplace=False)
        (4): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (14): ResnetBlock(
      (conv_block): Sequential(
        (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.0, inplace=False)
        (4): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (15): ConvTranspose2d(576, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (16): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (17): ReLU(inplace=True)
    (18): ConvTranspose2d(192, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (19): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (20): ReLU(inplace=True)
    (21): Conv2d(64, 26, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
    (22): Tanh()
  )
)
Total number of parameters: 38230746
ResnetEncoder(
  (model): Sequential(
    (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(64, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (4): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): Conv2d(192, 576, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (7): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU(inplace=True)
    (9): ResnetBlock(
      (conv_block): Sequential(
        (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.5, inplace=False)
        (4): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (10): ResnetBlock(
      (conv_block): Sequential(
        (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.5, inplace=False)
        (4): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (11): ResnetBlock(
      (conv_block): Sequential(
        (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.5, inplace=False)
        (4): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (12): ResnetBlock(
      (conv_block): Sequential(
        (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.5, inplace=False)
        (4): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (13): ResnetBlock(
      (conv_block): Sequential(
        (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.5, inplace=False)
        (4): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (14): ResnetBlock(
      (conv_block): Sequential(
        (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.5, inplace=False)
        (4): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
)
Total number of parameters: 36970368
ResnetDecoder(
  (model): Sequential(
    (0): ConvTranspose2d(576, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): ConvTranspose2d(192, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): Conv2d(64, 3, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
    (7): Tanh()
  )
)
Total number of parameters: 1116099
InputTransformation(
  (model): Sequential(
    (0): Conv2d(6, 6, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (1): BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(6, 6, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (4): BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
  )
)
Total number of parameters: 684
NLayerDiscriminator(
  (model): Sequential(
    (0): Conv2d(6, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
    (1): LeakyReLU(negative_slope=0.2, inplace=True)
    (2): Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): LeakyReLU(negative_slope=0.2, inplace=True)
    (5): Conv2d(128, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  )
)
Total number of parameters: 218049
-----------------------------------------------
model [StackGANModel] was created
create web directory ./checkpoints/ft37_1_MCGAN_train/web...
starting propagating back to the first network with starting lr 0.0002 ...
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
---------- Networks initialized -------------
ResnetGenerator_3d_conv(
  (model): Sequential(
    (0): Conv3d(26, 26, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=26)
    (1): BatchNorm3d(26, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
)
Total number of parameters: 780
ResnetGenerator(
  (model): Sequential(
    (0): Conv2d(26, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(64, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (4): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): Conv2d(192, 576, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (7): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU(inplace=True)
    (9): ResnetBlock(
      (conv_block): Sequential(
        (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.5, inplace=False)
        (4): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (10): ResnetBlock(
      (conv_block): Sequential(
        (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.5, inplace=False)
        (4): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (11): ResnetBlock(
      (conv_block): Sequential(
        (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.5, inplace=False)
        (4): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (12): ResnetBlock(
      (conv_block): Sequential(
        (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.5, inplace=False)
        (4): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (13): ResnetBlock(
      (conv_block): Sequential(
        (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.5, inplace=False)
        (4): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (14): ResnetBlock(
      (conv_block): Sequential(
        (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.5, inplace=False)
        (4): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (15): ConvTranspose2d(576, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (16): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (17): ReLU(inplace=True)
    (18): ConvTranspose2d(192, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (19): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (20): ReLU(inplace=True)
    (21): Conv2d(64, 26, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
    (22): Tanh()
  )
)
Total number of parameters: 38230746
ResnetEncoder(
  (model): Sequential(
    (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(64, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (4): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): Conv2d(192, 576, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (7): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU(inplace=True)
    (9): ResnetBlock(
      (conv_block): Sequential(
        (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.5, inplace=False)
        (4): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (10): ResnetBlock(
      (conv_block): Sequential(
        (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.5, inplace=False)
        (4): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (11): ResnetBlock(
      (conv_block): Sequential(
        (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.5, inplace=False)
        (4): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (12): ResnetBlock(
      (conv_block): Sequential(
        (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.5, inplace=False)
        (4): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (13): ResnetBlock(
      (conv_block): Sequential(
        (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.5, inplace=False)
        (4): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (14): ResnetBlock(
      (conv_block): Sequential(
        (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.5, inplace=False)
        (4): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
)
Total number of parameters: 36970368
ResnetDecoder(
  (model): Sequential(
    (0): ConvTranspose2d(576, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): ConvTranspose2d(192, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): Conv2d(64, 3, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
    (7): Tanh()
  )
)
Total number of parameters: 1116099
InputTransformation(
  (model): Sequential(
    (0): Conv2d(6, 6, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (1): BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(6, 6, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (4): BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
  )
)
Total number of parameters: 684
NLayerDiscriminator(
  (model): Sequential(
    (0): Conv2d(6, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
    (1): LeakyReLU(negative_slope=0.2, inplace=True)
    (2): Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): LeakyReLU(negative_slope=0.2, inplace=True)
    (5): Conv2d(128, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  )
)
Total number of parameters: 218049
-----------------------------------------------
model [StackGANModel] was created
create web directory ./checkpoints/ft37_1_MCGAN_train/web...
saving the model at the end of epoch 0, iters 0
End of epoch 1 / 700 	 Time Taken: 0 sec
/home/gt/.virtualenvs/pytorch/lib/python3.6/site-packages/torch/nn/functional.py:1350: UserWarning: nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
  warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")
End of epoch 2 / 700 	 Time Taken: 0 sec
End of epoch 3 / 700 	 Time Taken: 0 sec
End of epoch 4 / 700 	 Time Taken: 0 sec
(epoch: 5, iters: 15, time: 0.041) G1_GAN: 0.754 G1_L1: 125.258 G1_MSE_gt: 56.583 G1_MSE: 40.723 D1_real: 0.637 D1_fake: 0.995 G_L1: 4.773 
End of epoch 5 / 700 	 Time Taken: 0 sec
End of epoch 6 / 700 	 Time Taken: 0 sec
End of epoch 7 / 700 	 Time Taken: 0 sec
End of epoch 8 / 700 	 Time Taken: 0 sec
End of epoch 9 / 700 	 Time Taken: 0 sec
(epoch: 10, iters: 25, time: 0.040) G1_GAN: 0.798 G1_L1: 78.673 G1_MSE_gt: 47.138 G1_MSE: 11.288 D1_real: 0.586 D1_fake: 0.698 G_L1: 5.094 
End of epoch 10 / 700 	 Time Taken: 0 sec
End of epoch 11 / 700 	 Time Taken: 0 sec
End of epoch 12 / 700 	 Time Taken: 0 sec
End of epoch 13 / 700 	 Time Taken: 0 sec
End of epoch 14 / 700 	 Time Taken: 0 sec
(epoch: 15, iters: 35, time: 0.041) G1_GAN: 0.708 G1_L1: 69.049 G1_MSE_gt: 23.232 G1_MSE: 6.680 D1_real: 0.509 D1_fake: 0.609 G_L1: 5.435 
End of epoch 15 / 700 	 Time Taken: 0 sec
End of epoch 16 / 700 	 Time Taken: 0 sec
End of epoch 17 / 700 	 Time Taken: 0 sec
End of epoch 18 / 700 	 Time Taken: 0 sec
End of epoch 19 / 700 	 Time Taken: 0 sec
(epoch: 20, iters: 45, time: 0.041) G1_GAN: 0.693 G1_L1: 60.274 G1_MSE_gt: 11.023 G1_MSE: 5.261 D1_real: 0.487 D1_fake: 0.536 G_L1: 6.161 
End of epoch 20 / 700 	 Time Taken: 0 sec
End of epoch 21 / 700 	 Time Taken: 0 sec
End of epoch 22 / 700 	 Time Taken: 0 sec
End of epoch 23 / 700 	 Time Taken: 0 sec
End of epoch 24 / 700 	 Time Taken: 0 sec
(epoch: 25, iters: 55, time: 0.041) G1_GAN: 0.703 G1_L1: 51.031 G1_MSE_gt: 9.674 G1_MSE: 2.908 D1_real: 0.453 D1_fake: 0.527 G_L1: 5.778 
End of epoch 25 / 700 	 Time Taken: 0 sec
End of epoch 26 / 700 	 Time Taken: 0 sec
End of epoch 27 / 700 	 Time Taken: 0 sec
End of epoch 28 / 700 	 Time Taken: 0 sec
End of epoch 29 / 700 	 Time Taken: 0 sec
(epoch: 30, iters: 65, time: 0.041) G1_GAN: 0.707 G1_L1: 45.867 G1_MSE_gt: 7.756 G1_MSE: 3.526 D1_real: 0.447 D1_fake: 0.471 G_L1: 5.662 
End of epoch 30 / 700 	 Time Taken: 0 sec
End of epoch 31 / 700 	 Time Taken: 0 sec
End of epoch 32 / 700 	 Time Taken: 0 sec
End of epoch 33 / 700 	 Time Taken: 0 sec
End of epoch 34 / 700 	 Time Taken: 0 sec
(epoch: 35, iters: 75, time: 0.042) G1_GAN: 0.728 G1_L1: 44.692 G1_MSE_gt: 6.173 G1_MSE: 2.575 D1_real: 0.433 D1_fake: 0.499 G_L1: 5.422 
End of epoch 35 / 700 	 Time Taken: 0 sec
End of epoch 36 / 700 	 Time Taken: 0 sec
End of epoch 37 / 700 	 Time Taken: 0 sec
End of epoch 38 / 700 	 Time Taken: 0 sec
End of epoch 39 / 700 	 Time Taken: 0 sec
(epoch: 40, iters: 85, time: 0.041) G1_GAN: 0.766 G1_L1: 42.155 G1_MSE_gt: 5.639 G1_MSE: 3.299 D1_real: 0.393 D1_fake: 0.461 G_L1: 5.404 
End of epoch 40 / 700 	 Time Taken: 0 sec
End of epoch 41 / 700 	 Time Taken: 0 sec
End of epoch 42 / 700 	 Time Taken: 0 sec
End of epoch 43 / 700 	 Time Taken: 0 sec
End of epoch 44 / 700 	 Time Taken: 0 sec
(epoch: 45, iters: 95, time: 0.042) G1_GAN: 0.799 G1_L1: 37.589 G1_MSE_gt: 5.344 G1_MSE: 2.344 D1_real: 0.383 D1_fake: 0.449 G_L1: 5.350 
End of epoch 45 / 700 	 Time Taken: 0 sec
End of epoch 46 / 700 	 Time Taken: 0 sec
End of epoch 47 / 700 	 Time Taken: 0 sec
End of epoch 48 / 700 	 Time Taken: 0 sec
End of epoch 49 / 700 	 Time Taken: 0 sec
(epoch: 50, iters: 105, time: 0.043) G1_GAN: 0.851 G1_L1: 34.992 G1_MSE_gt: 5.385 G1_MSE: 2.263 D1_real: 0.500 D1_fake: 0.561 G_L1: 5.343 
End of epoch 50 / 700 	 Time Taken: 0 sec
End of epoch 51 / 700 	 Time Taken: 0 sec
End of epoch 52 / 700 	 Time Taken: 0 sec
End of epoch 53 / 700 	 Time Taken: 0 sec
End of epoch 54 / 700 	 Time Taken: 0 sec
(epoch: 55, iters: 115, time: 0.043) G1_GAN: 0.792 G1_L1: 33.144 G1_MSE_gt: 4.864 G1_MSE: 1.864 D1_real: 0.356 D1_fake: 0.401 G_L1: 5.227 
End of epoch 55 / 700 	 Time Taken: 0 sec
End of epoch 56 / 700 	 Time Taken: 0 sec
End of epoch 57 / 700 	 Time Taken: 0 sec
End of epoch 58 / 700 	 Time Taken: 0 sec
End of epoch 59 / 700 	 Time Taken: 0 sec
(epoch: 60, iters: 125, time: 0.043) G1_GAN: 0.834 G1_L1: 31.292 G1_MSE_gt: 3.504 G1_MSE: 2.361 D1_real: 0.360 D1_fake: 0.406 G_L1: 5.095 
End of epoch 60 / 700 	 Time Taken: 0 sec
End of epoch 61 / 700 	 Time Taken: 0 sec
End of epoch 62 / 700 	 Time Taken: 0 sec
End of epoch 63 / 700 	 Time Taken: 0 sec
End of epoch 64 / 700 	 Time Taken: 0 sec
(epoch: 65, iters: 135, time: 0.048) G1_GAN: 0.878 G1_L1: 28.527 G1_MSE_gt: 3.446 G1_MSE: 1.841 D1_real: 0.329 D1_fake: 0.409 G_L1: 5.119 
End of epoch 65 / 700 	 Time Taken: 0 sec
End of epoch 66 / 700 	 Time Taken: 0 sec
End of epoch 67 / 700 	 Time Taken: 0 sec
End of epoch 68 / 700 	 Time Taken: 0 sec
End of epoch 69 / 700 	 Time Taken: 0 sec
(epoch: 70, iters: 145, time: 0.043) G1_GAN: 0.852 G1_L1: 27.902 G1_MSE_gt: 3.282 G1_MSE: 1.827 D1_real: 0.298 D1_fake: 0.379 G_L1: 5.072 
End of epoch 70 / 700 	 Time Taken: 0 sec
End of epoch 71 / 700 	 Time Taken: 0 sec
End of epoch 72 / 700 	 Time Taken: 0 sec
End of epoch 73 / 700 	 Time Taken: 0 sec
End of epoch 74 / 700 	 Time Taken: 0 sec
(epoch: 75, iters: 155, time: 0.046) G1_GAN: 0.832 G1_L1: 28.767 G1_MSE_gt: 3.644 G1_MSE: 2.122 D1_real: 0.326 D1_fake: 0.359 G_L1: 5.253 
End of epoch 75 / 700 	 Time Taken: 0 sec
End of epoch 76 / 700 	 Time Taken: 0 sec
End of epoch 77 / 700 	 Time Taken: 0 sec
End of epoch 78 / 700 	 Time Taken: 0 sec
End of epoch 79 / 700 	 Time Taken: 0 sec
(epoch: 80, iters: 165, time: 0.044) G1_GAN: 0.936 G1_L1: 26.005 G1_MSE_gt: 2.866 G1_MSE: 1.909 D1_real: 0.386 D1_fake: 0.371 G_L1: 4.945 
End of epoch 80 / 700 	 Time Taken: 0 sec
End of epoch 81 / 700 	 Time Taken: 0 sec
End of epoch 82 / 700 	 Time Taken: 0 sec
End of epoch 83 / 700 	 Time Taken: 0 sec
End of epoch 84 / 700 	 Time Taken: 0 sec
(epoch: 85, iters: 175, time: 0.044) G1_GAN: 0.937 G1_L1: 24.615 G1_MSE_gt: 2.864 G1_MSE: 1.820 D1_real: 0.292 D1_fake: 0.334 G_L1: 4.854 
End of epoch 85 / 700 	 Time Taken: 0 sec
End of epoch 86 / 700 	 Time Taken: 0 sec
End of epoch 87 / 700 	 Time Taken: 0 sec
End of epoch 88 / 700 	 Time Taken: 0 sec
End of epoch 89 / 700 	 Time Taken: 0 sec
(epoch: 90, iters: 185, time: 0.044) G1_GAN: 0.983 G1_L1: 23.364 G1_MSE_gt: 2.240 G1_MSE: 1.938 D1_real: 0.325 D1_fake: 0.355 G_L1: 4.845 
End of epoch 90 / 700 	 Time Taken: 0 sec
End of epoch 91 / 700 	 Time Taken: 0 sec
End of epoch 92 / 700 	 Time Taken: 0 sec
End of epoch 93 / 700 	 Time Taken: 0 sec
End of epoch 94 / 700 	 Time Taken: 0 sec
(epoch: 95, iters: 195, time: 0.043) G1_GAN: 1.059 G1_L1: 21.617 G1_MSE_gt: 2.204 G1_MSE: 1.668 D1_real: 0.337 D1_fake: 0.476 G_L1: 4.806 
End of epoch 95 / 700 	 Time Taken: 0 sec
End of epoch 96 / 700 	 Time Taken: 0 sec
End of epoch 97 / 700 	 Time Taken: 0 sec
End of epoch 98 / 700 	 Time Taken: 0 sec
End of epoch 99 / 700 	 Time Taken: 0 sec
(epoch: 100, iters: 205, time: 0.044) G1_GAN: 0.982 G1_L1: 21.578 G1_MSE_gt: 2.008 G1_MSE: 1.892 D1_real: 0.300 D1_fake: 0.297 G_L1: 4.765 
saving the model at the end of epoch 100, iters 700
End of epoch 100 / 700 	 Time Taken: 1 sec
End of epoch 101 / 700 	 Time Taken: 0 sec
End of epoch 102 / 700 	 Time Taken: 0 sec
End of epoch 103 / 700 	 Time Taken: 0 sec
End of epoch 104 / 700 	 Time Taken: 0 sec
(epoch: 105, iters: 215, time: 0.044) G1_GAN: 1.072 G1_L1: 21.522 G1_MSE_gt: 2.572 G1_MSE: 1.839 D1_real: 0.238 D1_fake: 0.397 G_L1: 4.752 
End of epoch 105 / 700 	 Time Taken: 0 sec
End of epoch 106 / 700 	 Time Taken: 0 sec
End of epoch 107 / 700 	 Time Taken: 0 sec
End of epoch 108 / 700 	 Time Taken: 0 sec
End of epoch 109 / 700 	 Time Taken: 0 sec
(epoch: 110, iters: 225, time: 0.053) G1_GAN: 1.105 G1_L1: 20.837 G1_MSE_gt: 2.464 G1_MSE: 1.493 D1_real: 0.245 D1_fake: 0.372 G_L1: 4.635 
End of epoch 110 / 700 	 Time Taken: 0 sec
End of epoch 111 / 700 	 Time Taken: 0 sec
End of epoch 112 / 700 	 Time Taken: 0 sec
End of epoch 113 / 700 	 Time Taken: 0 sec
End of epoch 114 / 700 	 Time Taken: 0 sec
(epoch: 115, iters: 235, time: 0.045) G1_GAN: 0.893 G1_L1: 19.242 G1_MSE_gt: 1.694 G1_MSE: 1.741 D1_real: 0.288 D1_fake: 0.251 G_L1: 4.705 
End of epoch 115 / 700 	 Time Taken: 0 sec
End of epoch 116 / 700 	 Time Taken: 0 sec
End of epoch 117 / 700 	 Time Taken: 0 sec
End of epoch 118 / 700 	 Time Taken: 0 sec
End of epoch 119 / 700 	 Time Taken: 0 sec
(epoch: 120, iters: 245, time: 0.045) G1_GAN: 1.153 G1_L1: 19.382 G1_MSE_gt: 1.564 G1_MSE: 1.722 D1_real: 0.239 D1_fake: 0.333 G_L1: 4.606 
End of epoch 120 / 700 	 Time Taken: 0 sec
End of epoch 121 / 700 	 Time Taken: 0 sec
End of epoch 122 / 700 	 Time Taken: 0 sec
End of epoch 123 / 700 	 Time Taken: 0 sec
End of epoch 124 / 700 	 Time Taken: 0 sec
(epoch: 125, iters: 255, time: 0.045) G1_GAN: 0.981 G1_L1: 18.900 G1_MSE_gt: 1.437 G1_MSE: 1.822 D1_real: 0.370 D1_fake: 0.240 G_L1: 4.644 
End of epoch 125 / 700 	 Time Taken: 0 sec
End of epoch 126 / 700 	 Time Taken: 0 sec
End of epoch 127 / 700 	 Time Taken: 0 sec
End of epoch 128 / 700 	 Time Taken: 0 sec
End of epoch 129 / 700 	 Time Taken: 0 sec
(epoch: 130, iters: 265, time: 0.045) G1_GAN: 1.057 G1_L1: 17.127 G1_MSE_gt: 1.670 G1_MSE: 1.626 D1_real: 0.215 D1_fake: 0.287 G_L1: 4.554 
End of epoch 130 / 700 	 Time Taken: 0 sec
End of epoch 131 / 700 	 Time Taken: 0 sec
End of epoch 132 / 700 	 Time Taken: 0 sec
End of epoch 133 / 700 	 Time Taken: 0 sec
End of epoch 134 / 700 	 Time Taken: 0 sec
(epoch: 135, iters: 275, time: 0.046) G1_GAN: 1.067 G1_L1: 18.444 G1_MSE_gt: 1.326 G1_MSE: 1.575 D1_real: 0.261 D1_fake: 0.258 G_L1: 4.468 
End of epoch 135 / 700 	 Time Taken: 0 sec
End of epoch 136 / 700 	 Time Taken: 0 sec
End of epoch 137 / 700 	 Time Taken: 0 sec
End of epoch 138 / 700 	 Time Taken: 0 sec
End of epoch 139 / 700 	 Time Taken: 0 sec
(epoch: 140, iters: 285, time: 0.054) G1_GAN: 1.216 G1_L1: 16.100 G1_MSE_gt: 1.389 G1_MSE: 1.486 D1_real: 0.258 D1_fake: 0.324 G_L1: 4.517 
End of epoch 140 / 700 	 Time Taken: 0 sec
End of epoch 141 / 700 	 Time Taken: 0 sec
End of epoch 142 / 700 	 Time Taken: 0 sec
End of epoch 143 / 700 	 Time Taken: 0 sec
End of epoch 144 / 700 	 Time Taken: 0 sec
(epoch: 145, iters: 295, time: 0.046) G1_GAN: 1.065 G1_L1: 15.895 G1_MSE_gt: 1.211 G1_MSE: 1.533 D1_real: 0.272 D1_fake: 0.263 G_L1: 4.492 
End of epoch 145 / 700 	 Time Taken: 0 sec
End of epoch 146 / 700 	 Time Taken: 0 sec
End of epoch 147 / 700 	 Time Taken: 0 sec
End of epoch 148 / 700 	 Time Taken: 0 sec
End of epoch 149 / 700 	 Time Taken: 0 sec
(epoch: 150, iters: 305, time: 0.046) G1_GAN: 1.468 G1_L1: 16.543 G1_MSE_gt: 1.331 G1_MSE: 1.499 D1_real: 0.229 D1_fake: 0.306 G_L1: 4.389 
End of epoch 150 / 700 	 Time Taken: 0 sec
End of epoch 151 / 700 	 Time Taken: 0 sec
End of epoch 152 / 700 	 Time Taken: 0 sec
End of epoch 153 / 700 	 Time Taken: 0 sec
End of epoch 154 / 700 	 Time Taken: 0 sec
(epoch: 155, iters: 315, time: 0.045) G1_GAN: 1.167 G1_L1: 15.823 G1_MSE_gt: 1.102 G1_MSE: 1.522 D1_real: 0.223 D1_fake: 0.262 G_L1: 4.389 
End of epoch 155 / 700 	 Time Taken: 0 sec
End of epoch 156 / 700 	 Time Taken: 0 sec
End of epoch 157 / 700 	 Time Taken: 0 sec
End of epoch 158 / 700 	 Time Taken: 0 sec
End of epoch 159 / 700 	 Time Taken: 0 sec
(epoch: 160, iters: 325, time: 0.046) G1_GAN: 1.147 G1_L1: 15.458 G1_MSE_gt: 1.180 G1_MSE: 1.513 D1_real: 0.225 D1_fake: 0.241 G_L1: 4.433 
End of epoch 160 / 700 	 Time Taken: 0 sec
End of epoch 161 / 700 	 Time Taken: 0 sec
End of epoch 162 / 700 	 Time Taken: 0 sec
End of epoch 163 / 700 	 Time Taken: 0 sec
End of epoch 164 / 700 	 Time Taken: 0 sec
(epoch: 165, iters: 335, time: 0.055) G1_GAN: 1.381 G1_L1: 14.005 G1_MSE_gt: 1.162 G1_MSE: 1.318 D1_real: 0.279 D1_fake: 0.290 G_L1: 4.394 
End of epoch 165 / 700 	 Time Taken: 0 sec
End of epoch 166 / 700 	 Time Taken: 0 sec
End of epoch 167 / 700 	 Time Taken: 0 sec
End of epoch 168 / 700 	 Time Taken: 0 sec
End of epoch 169 / 700 	 Time Taken: 0 sec
(epoch: 170, iters: 345, time: 0.046) G1_GAN: 1.142 G1_L1: 15.648 G1_MSE_gt: 1.184 G1_MSE: 1.578 D1_real: 0.259 D1_fake: 0.256 G_L1: 4.300 
End of epoch 170 / 700 	 Time Taken: 0 sec
End of epoch 171 / 700 	 Time Taken: 0 sec
End of epoch 172 / 700 	 Time Taken: 0 sec
End of epoch 173 / 700 	 Time Taken: 0 sec
End of epoch 174 / 700 	 Time Taken: 0 sec
(epoch: 175, iters: 355, time: 0.045) G1_GAN: 1.070 G1_L1: 14.014 G1_MSE_gt: 1.090 G1_MSE: 1.357 D1_real: 0.254 D1_fake: 0.217 G_L1: 4.257 
End of epoch 175 / 700 	 Time Taken: 0 sec
End of epoch 176 / 700 	 Time Taken: 0 sec
End of epoch 177 / 700 	 Time Taken: 0 sec
End of epoch 178 / 700 	 Time Taken: 0 sec
End of epoch 179 / 700 	 Time Taken: 0 sec
(epoch: 180, iters: 365, time: 0.049) G1_GAN: 1.591 G1_L1: 15.781 G1_MSE_gt: 1.037 G1_MSE: 1.480 D1_real: 0.178 D1_fake: 0.368 G_L1: 4.227 
End of epoch 180 / 700 	 Time Taken: 0 sec
End of epoch 181 / 700 	 Time Taken: 0 sec
End of epoch 182 / 700 	 Time Taken: 0 sec
End of epoch 183 / 700 	 Time Taken: 0 sec
End of epoch 184 / 700 	 Time Taken: 0 sec
(epoch: 185, iters: 375, time: 0.054) G1_GAN: 0.959 G1_L1: 14.527 G1_MSE_gt: 0.833 G1_MSE: 1.720 D1_real: 0.304 D1_fake: 0.210 G_L1: 4.276 
End of epoch 185 / 700 	 Time Taken: 0 sec
End of epoch 186 / 700 	 Time Taken: 0 sec
End of epoch 187 / 700 	 Time Taken: 0 sec
End of epoch 188 / 700 	 Time Taken: 0 sec
End of epoch 189 / 700 	 Time Taken: 0 sec
(epoch: 190, iters: 385, time: 0.046) G1_GAN: 1.476 G1_L1: 13.410 G1_MSE_gt: 1.294 G1_MSE: 1.411 D1_real: 0.180 D1_fake: 0.276 G_L1: 4.159 
End of epoch 190 / 700 	 Time Taken: 0 sec
End of epoch 191 / 700 	 Time Taken: 0 sec
End of epoch 192 / 700 	 Time Taken: 0 sec
End of epoch 193 / 700 	 Time Taken: 0 sec
End of epoch 194 / 700 	 Time Taken: 0 sec
(epoch: 195, iters: 395, time: 0.047) G1_GAN: 1.258 G1_L1: 13.277 G1_MSE_gt: 0.900 G1_MSE: 1.588 D1_real: 0.222 D1_fake: 0.206 G_L1: 4.215 
End of epoch 195 / 700 	 Time Taken: 0 sec
End of epoch 196 / 700 	 Time Taken: 0 sec
End of epoch 197 / 700 	 Time Taken: 0 sec
End of epoch 198 / 700 	 Time Taken: 0 sec
End of epoch 199 / 700 	 Time Taken: 0 sec
(epoch: 200, iters: 405, time: 0.047) G1_GAN: 1.157 G1_L1: 13.416 G1_MSE_gt: 0.916 G1_MSE: 1.476 D1_real: 0.199 D1_fake: 0.221 G_L1: 4.216 
saving the model at the end of epoch 200, iters 1400
End of epoch 200 / 700 	 Time Taken: 1 sec
End of epoch 201 / 700 	 Time Taken: 0 sec
End of epoch 202 / 700 	 Time Taken: 0 sec
End of epoch 203 / 700 	 Time Taken: 0 sec
End of epoch 204 / 700 	 Time Taken: 0 sec
(epoch: 205, iters: 415, time: 0.056) G1_GAN: 1.207 G1_L1: 12.223 G1_MSE_gt: 0.859 G1_MSE: 1.318 D1_real: 0.257 D1_fake: 0.243 G_L1: 4.117 
End of epoch 205 / 700 	 Time Taken: 0 sec
End of epoch 206 / 700 	 Time Taken: 0 sec
End of epoch 207 / 700 	 Time Taken: 0 sec
End of epoch 208 / 700 	 Time Taken: 0 sec
End of epoch 209 / 700 	 Time Taken: 0 sec
(epoch: 210, iters: 425, time: 0.048) G1_GAN: 1.299 G1_L1: 12.243 G1_MSE_gt: 0.850 G1_MSE: 1.232 D1_real: 0.229 D1_fake: 0.259 G_L1: 4.101 
End of epoch 210 / 700 	 Time Taken: 0 sec
End of epoch 211 / 700 	 Time Taken: 0 sec
End of epoch 212 / 700 	 Time Taken: 0 sec
End of epoch 213 / 700 	 Time Taken: 0 sec
End of epoch 214 / 700 	 Time Taken: 0 sec
(epoch: 215, iters: 435, time: 0.048) G1_GAN: 1.515 G1_L1: 11.901 G1_MSE_gt: 0.860 G1_MSE: 1.443 D1_real: 0.268 D1_fake: 0.223 G_L1: 4.120 
End of epoch 215 / 700 	 Time Taken: 0 sec
End of epoch 216 / 700 	 Time Taken: 0 sec
End of epoch 217 / 700 	 Time Taken: 0 sec
End of epoch 218 / 700 	 Time Taken: 0 sec
End of epoch 219 / 700 	 Time Taken: 0 sec
(epoch: 220, iters: 445, time: 0.047) G1_GAN: 1.108 G1_L1: 12.173 G1_MSE_gt: 0.752 G1_MSE: 1.251 D1_real: 0.219 D1_fake: 0.185 G_L1: 4.056 
End of epoch 220 / 700 	 Time Taken: 0 sec
End of epoch 221 / 700 	 Time Taken: 0 sec
End of epoch 222 / 700 	 Time Taken: 0 sec
End of epoch 223 / 700 	 Time Taken: 0 sec
End of epoch 224 / 700 	 Time Taken: 0 sec
(epoch: 225, iters: 455, time: 0.056) G1_GAN: 1.366 G1_L1: 11.700 G1_MSE_gt: 0.700 G1_MSE: 1.437 D1_real: 0.289 D1_fake: 0.203 G_L1: 4.142 
End of epoch 225 / 700 	 Time Taken: 0 sec
End of epoch 226 / 700 	 Time Taken: 0 sec
End of epoch 227 / 700 	 Time Taken: 0 sec
End of epoch 228 / 700 	 Time Taken: 0 sec
End of epoch 229 / 700 	 Time Taken: 0 sec
(epoch: 230, iters: 465, time: 0.048) G1_GAN: 1.393 G1_L1: 11.672 G1_MSE_gt: 0.715 G1_MSE: 1.399 D1_real: 0.168 D1_fake: 0.256 G_L1: 4.001 
End of epoch 230 / 700 	 Time Taken: 0 sec
End of epoch 231 / 700 	 Time Taken: 0 sec
End of epoch 232 / 700 	 Time Taken: 0 sec
End of epoch 233 / 700 	 Time Taken: 0 sec
End of epoch 234 / 700 	 Time Taken: 0 sec
(epoch: 235, iters: 475, time: 0.048) G1_GAN: 1.305 G1_L1: 11.345 G1_MSE_gt: 0.685 G1_MSE: 1.435 D1_real: 0.143 D1_fake: 0.274 G_L1: 4.009 
End of epoch 235 / 700 	 Time Taken: 0 sec
End of epoch 236 / 700 	 Time Taken: 0 sec
End of epoch 237 / 700 	 Time Taken: 0 sec
End of epoch 238 / 700 	 Time Taken: 0 sec
End of epoch 239 / 700 	 Time Taken: 0 sec
(epoch: 240, iters: 485, time: 0.056) G1_GAN: 1.343 G1_L1: 11.965 G1_MSE_gt: 0.774 G1_MSE: 1.369 D1_real: 0.198 D1_fake: 0.210 G_L1: 3.931 
End of epoch 240 / 700 	 Time Taken: 0 sec
End of epoch 241 / 700 	 Time Taken: 0 sec
End of epoch 242 / 700 	 Time Taken: 0 sec
End of epoch 243 / 700 	 Time Taken: 0 sec
End of epoch 244 / 700 	 Time Taken: 0 sec
(epoch: 245, iters: 495, time: 0.048) G1_GAN: 1.391 G1_L1: 11.125 G1_MSE_gt: 0.722 G1_MSE: 1.527 D1_real: 0.177 D1_fake: 0.314 G_L1: 3.960 
End of epoch 245 / 700 	 Time Taken: 0 sec
End of epoch 246 / 700 	 Time Taken: 0 sec
End of epoch 247 / 700 	 Time Taken: 0 sec
End of epoch 248 / 700 	 Time Taken: 0 sec
End of epoch 249 / 700 	 Time Taken: 0 sec
(epoch: 250, iters: 505, time: 0.048) G1_GAN: 1.181 G1_L1: 10.841 G1_MSE_gt: 0.747 G1_MSE: 1.323 D1_real: 0.251 D1_fake: 0.215 G_L1: 3.911 
End of epoch 250 / 700 	 Time Taken: 0 sec
End of epoch 251 / 700 	 Time Taken: 0 sec
End of epoch 252 / 700 	 Time Taken: 0 sec
End of epoch 253 / 700 	 Time Taken: 0 sec
End of epoch 254 / 700 	 Time Taken: 0 sec
(epoch: 255, iters: 515, time: 0.057) G1_GAN: 1.325 G1_L1: 10.800 G1_MSE_gt: 0.803 G1_MSE: 1.425 D1_real: 0.166 D1_fake: 0.234 G_L1: 3.922 
End of epoch 255 / 700 	 Time Taken: 0 sec
End of epoch 256 / 700 	 Time Taken: 0 sec
End of epoch 257 / 700 	 Time Taken: 0 sec
End of epoch 258 / 700 	 Time Taken: 0 sec
End of epoch 259 / 700 	 Time Taken: 0 sec
(epoch: 260, iters: 525, time: 0.049) G1_GAN: 1.145 G1_L1: 10.671 G1_MSE_gt: 0.783 G1_MSE: 1.397 D1_real: 0.210 D1_fake: 0.183 G_L1: 3.894 
End of epoch 260 / 700 	 Time Taken: 0 sec
End of epoch 261 / 700 	 Time Taken: 0 sec
End of epoch 262 / 700 	 Time Taken: 0 sec
End of epoch 263 / 700 	 Time Taken: 0 sec
End of epoch 264 / 700 	 Time Taken: 0 sec
(epoch: 265, iters: 535, time: 0.049) G1_GAN: 1.245 G1_L1: 10.348 G1_MSE_gt: 0.675 G1_MSE: 1.479 D1_real: 0.188 D1_fake: 0.181 G_L1: 3.947 
End of epoch 265 / 700 	 Time Taken: 0 sec
End of epoch 266 / 700 	 Time Taken: 0 sec
End of epoch 267 / 700 	 Time Taken: 0 sec
End of epoch 268 / 700 	 Time Taken: 0 sec
End of epoch 269 / 700 	 Time Taken: 0 sec
(epoch: 270, iters: 545, time: 0.057) G1_GAN: 1.270 G1_L1: 10.257 G1_MSE_gt: 0.662 G1_MSE: 1.377 D1_real: 0.172 D1_fake: 0.215 G_L1: 3.811 
End of epoch 270 / 700 	 Time Taken: 0 sec
End of epoch 271 / 700 	 Time Taken: 0 sec
End of epoch 272 / 700 	 Time Taken: 0 sec
End of epoch 273 / 700 	 Time Taken: 0 sec
End of epoch 274 / 700 	 Time Taken: 0 sec
(epoch: 275, iters: 555, time: 0.050) G1_GAN: 1.228 G1_L1: 10.572 G1_MSE_gt: 0.676 G1_MSE: 1.439 D1_real: 0.185 D1_fake: 0.174 G_L1: 3.822 
End of epoch 275 / 700 	 Time Taken: 0 sec
End of epoch 276 / 700 	 Time Taken: 0 sec
End of epoch 277 / 700 	 Time Taken: 0 sec
End of epoch 278 / 700 	 Time Taken: 0 sec
End of epoch 279 / 700 	 Time Taken: 0 sec
(epoch: 280, iters: 565, time: 0.050) G1_GAN: 1.304 G1_L1: 9.714 G1_MSE_gt: 0.595 G1_MSE: 1.599 D1_real: 0.262 D1_fake: 0.160 G_L1: 3.850 
End of epoch 280 / 700 	 Time Taken: 0 sec
End of epoch 281 / 700 	 Time Taken: 0 sec
End of epoch 282 / 700 	 Time Taken: 0 sec
End of epoch 283 / 700 	 Time Taken: 0 sec
End of epoch 284 / 700 	 Time Taken: 0 sec
(epoch: 285, iters: 575, time: 0.059) G1_GAN: 1.541 G1_L1: 10.447 G1_MSE_gt: 0.777 G1_MSE: 1.393 D1_real: 0.214 D1_fake: 0.376 G_L1: 3.811 
End of epoch 285 / 700 	 Time Taken: 0 sec
End of epoch 286 / 700 	 Time Taken: 0 sec
End of epoch 287 / 700 	 Time Taken: 0 sec
End of epoch 288 / 700 	 Time Taken: 0 sec
End of epoch 289 / 700 	 Time Taken: 0 sec
(epoch: 290, iters: 585, time: 0.050) G1_GAN: 1.234 G1_L1: 9.812 G1_MSE_gt: 0.682 G1_MSE: 1.668 D1_real: 0.279 D1_fake: 0.158 G_L1: 3.747 
End of epoch 290 / 700 	 Time Taken: 0 sec
End of epoch 291 / 700 	 Time Taken: 0 sec
End of epoch 292 / 700 	 Time Taken: 0 sec
End of epoch 293 / 700 	 Time Taken: 0 sec
End of epoch 294 / 700 	 Time Taken: 0 sec
(epoch: 295, iters: 595, time: 0.058) G1_GAN: 1.200 G1_L1: 9.765 G1_MSE_gt: 0.609 G1_MSE: 1.289 D1_real: 0.216 D1_fake: 0.155 G_L1: 3.808 
End of epoch 295 / 700 	 Time Taken: 0 sec
End of epoch 296 / 700 	 Time Taken: 0 sec
End of epoch 297 / 700 	 Time Taken: 0 sec
End of epoch 298 / 700 	 Time Taken: 0 sec
End of epoch 299 / 700 	 Time Taken: 0 sec
(epoch: 300, iters: 605, time: 0.051) G1_GAN: 1.163 G1_L1: 10.110 G1_MSE_gt: 0.581 G1_MSE: 1.438 D1_real: 0.147 D1_fake: 0.191 G_L1: 3.783 
saving the model at the end of epoch 300, iters 2100
End of epoch 300 / 700 	 Time Taken: 1 sec
End of epoch 301 / 700 	 Time Taken: 0 sec
End of epoch 302 / 700 	 Time Taken: 0 sec
End of epoch 303 / 700 	 Time Taken: 0 sec
End of epoch 304 / 700 	 Time Taken: 0 sec
(epoch: 305, iters: 615, time: 0.051) G1_GAN: 1.266 G1_L1: 9.877 G1_MSE_gt: 0.920 G1_MSE: 1.492 D1_real: 0.163 D1_fake: 0.195 G_L1: 3.845 
End of epoch 305 / 700 	 Time Taken: 0 sec
End of epoch 306 / 700 	 Time Taken: 0 sec
End of epoch 307 / 700 	 Time Taken: 0 sec
End of epoch 308 / 700 	 Time Taken: 0 sec
End of epoch 309 / 700 	 Time Taken: 0 sec
(epoch: 310, iters: 625, time: 0.059) G1_GAN: 1.590 G1_L1: 9.664 G1_MSE_gt: 0.635 G1_MSE: 1.540 D1_real: 0.237 D1_fake: 0.234 G_L1: 3.704 
End of epoch 310 / 700 	 Time Taken: 0 sec
End of epoch 311 / 700 	 Time Taken: 0 sec
End of epoch 312 / 700 	 Time Taken: 0 sec
End of epoch 313 / 700 	 Time Taken: 0 sec
End of epoch 314 / 700 	 Time Taken: 0 sec
(epoch: 315, iters: 635, time: 0.051) G1_GAN: 1.214 G1_L1: 9.580 G1_MSE_gt: 0.607 G1_MSE: 1.632 D1_real: 0.214 D1_fake: 0.183 G_L1: 3.691 
End of epoch 315 / 700 	 Time Taken: 0 sec
End of epoch 316 / 700 	 Time Taken: 0 sec
End of epoch 317 / 700 	 Time Taken: 0 sec
End of epoch 318 / 700 	 Time Taken: 0 sec
End of epoch 319 / 700 	 Time Taken: 0 sec
(epoch: 320, iters: 645, time: 0.058) G1_GAN: 1.152 G1_L1: 9.570 G1_MSE_gt: 0.596 G1_MSE: 1.347 D1_real: 0.208 D1_fake: 0.178 G_L1: 3.746 
End of epoch 320 / 700 	 Time Taken: 0 sec
End of epoch 321 / 700 	 Time Taken: 0 sec
End of epoch 322 / 700 	 Time Taken: 0 sec
End of epoch 323 / 700 	 Time Taken: 0 sec
End of epoch 324 / 700 	 Time Taken: 0 sec
(epoch: 325, iters: 655, time: 0.052) G1_GAN: 1.267 G1_L1: 9.571 G1_MSE_gt: 0.553 G1_MSE: 1.374 D1_real: 0.187 D1_fake: 0.182 G_L1: 3.663 
End of epoch 325 / 700 	 Time Taken: 0 sec
End of epoch 326 / 700 	 Time Taken: 0 sec
End of epoch 327 / 700 	 Time Taken: 0 sec
End of epoch 328 / 700 	 Time Taken: 0 sec
End of epoch 329 / 700 	 Time Taken: 0 sec
(epoch: 330, iters: 665, time: 0.051) G1_GAN: 1.590 G1_L1: 8.935 G1_MSE_gt: 0.681 G1_MSE: 1.363 D1_real: 0.131 D1_fake: 0.267 G_L1: 3.630 
End of epoch 330 / 700 	 Time Taken: 0 sec
End of epoch 331 / 700 	 Time Taken: 0 sec
End of epoch 332 / 700 	 Time Taken: 0 sec
End of epoch 333 / 700 	 Time Taken: 0 sec
End of epoch 334 / 700 	 Time Taken: 0 sec
(epoch: 335, iters: 675, time: 0.061) G1_GAN: 1.340 G1_L1: 10.167 G1_MSE_gt: 0.627 G1_MSE: 1.744 D1_real: 0.199 D1_fake: 0.152 G_L1: 3.621 
End of epoch 335 / 700 	 Time Taken: 0 sec
End of epoch 336 / 700 	 Time Taken: 0 sec
End of epoch 337 / 700 	 Time Taken: 0 sec
End of epoch 338 / 700 	 Time Taken: 0 sec
End of epoch 339 / 700 	 Time Taken: 0 sec
(epoch: 340, iters: 685, time: 0.052) G1_GAN: 1.260 G1_L1: 9.142 G1_MSE_gt: 0.596 G1_MSE: 1.593 D1_real: 0.143 D1_fake: 0.197 G_L1: 3.632 
End of epoch 340 / 700 	 Time Taken: 0 sec
End of epoch 341 / 700 	 Time Taken: 0 sec
End of epoch 342 / 700 	 Time Taken: 0 sec
End of epoch 343 / 700 	 Time Taken: 0 sec
End of epoch 344 / 700 	 Time Taken: 0 sec
(epoch: 345, iters: 695, time: 0.059) G1_GAN: 1.287 G1_L1: 8.752 G1_MSE_gt: 0.661 G1_MSE: 1.845 D1_real: 0.174 D1_fake: 0.180 G_L1: 3.675 
End of epoch 345 / 700 	 Time Taken: 0 sec
End of epoch 346 / 700 	 Time Taken: 0 sec
End of epoch 347 / 700 	 Time Taken: 0 sec
End of epoch 348 / 700 	 Time Taken: 0 sec
End of epoch 349 / 700 	 Time Taken: 0 sec
(epoch: 350, iters: 705, time: 0.052) G1_GAN: 1.316 G1_L1: 9.298 G1_MSE_gt: 0.644 G1_MSE: 1.345 D1_real: 0.214 D1_fake: 0.160 G_L1: 3.622 
End of epoch 350 / 700 	 Time Taken: 0 sec
End of epoch 351 / 700 	 Time Taken: 0 sec
End of epoch 352 / 700 	 Time Taken: 0 sec
End of epoch 353 / 700 	 Time Taken: 0 sec
End of epoch 354 / 700 	 Time Taken: 0 sec
(epoch: 355, iters: 715, time: 0.059) G1_GAN: 1.605 G1_L1: 9.469 G1_MSE_gt: 0.546 G1_MSE: 1.511 D1_real: 0.214 D1_fake: 0.524 G_L1: 3.576 
End of epoch 355 / 700 	 Time Taken: 0 sec
End of epoch 356 / 700 	 Time Taken: 0 sec
End of epoch 357 / 700 	 Time Taken: 0 sec
End of epoch 358 / 700 	 Time Taken: 0 sec
End of epoch 359 / 700 	 Time Taken: 0 sec
(epoch: 360, iters: 725, time: 0.053) G1_GAN: 1.428 G1_L1: 8.221 G1_MSE_gt: 0.513 G1_MSE: 1.610 D1_real: 0.159 D1_fake: 0.216 G_L1: 3.552 
End of epoch 360 / 700 	 Time Taken: 0 sec
End of epoch 361 / 700 	 Time Taken: 0 sec
End of epoch 362 / 700 	 Time Taken: 0 sec
End of epoch 363 / 700 	 Time Taken: 0 sec
End of epoch 364 / 700 	 Time Taken: 0 sec
(epoch: 365, iters: 735, time: 0.061) G1_GAN: 1.339 G1_L1: 8.446 G1_MSE_gt: 0.632 G1_MSE: 1.648 D1_real: 0.138 D1_fake: 0.153 G_L1: 3.542 
End of epoch 365 / 700 	 Time Taken: 0 sec
End of epoch 366 / 700 	 Time Taken: 0 sec
End of epoch 367 / 700 	 Time Taken: 0 sec
End of epoch 368 / 700 	 Time Taken: 0 sec
End of epoch 369 / 700 	 Time Taken: 0 sec
(epoch: 370, iters: 745, time: 0.053) G1_GAN: 1.672 G1_L1: 8.816 G1_MSE_gt: 0.622 G1_MSE: 1.917 D1_real: 0.113 D1_fake: 0.278 G_L1: 3.504 
End of epoch 370 / 700 	 Time Taken: 0 sec
End of epoch 371 / 700 	 Time Taken: 0 sec
End of epoch 372 / 700 	 Time Taken: 0 sec
End of epoch 373 / 700 	 Time Taken: 0 sec
End of epoch 374 / 700 	 Time Taken: 0 sec
(epoch: 375, iters: 755, time: 0.061) G1_GAN: 1.775 G1_L1: 8.647 G1_MSE_gt: 0.587 G1_MSE: 1.678 D1_real: 0.164 D1_fake: 0.196 G_L1: 3.448 
End of epoch 375 / 700 	 Time Taken: 0 sec
End of epoch 376 / 700 	 Time Taken: 0 sec
End of epoch 377 / 700 	 Time Taken: 0 sec
End of epoch 378 / 700 	 Time Taken: 0 sec
End of epoch 379 / 700 	 Time Taken: 0 sec
(epoch: 380, iters: 765, time: 0.053) G1_GAN: 1.311 G1_L1: 8.139 G1_MSE_gt: 0.575 G1_MSE: 1.696 D1_real: 0.166 D1_fake: 0.114 G_L1: 3.502 
End of epoch 380 / 700 	 Time Taken: 0 sec
End of epoch 381 / 700 	 Time Taken: 0 sec
End of epoch 382 / 700 	 Time Taken: 0 sec
End of epoch 383 / 700 	 Time Taken: 0 sec
End of epoch 384 / 700 	 Time Taken: 0 sec
(epoch: 385, iters: 775, time: 0.062) G1_GAN: 1.480 G1_L1: 8.700 G1_MSE_gt: 0.594 G1_MSE: 1.291 D1_real: 0.137 D1_fake: 0.197 G_L1: 3.473 
End of epoch 385 / 700 	 Time Taken: 0 sec
End of epoch 386 / 700 	 Time Taken: 0 sec
End of epoch 387 / 700 	 Time Taken: 0 sec
End of epoch 388 / 700 	 Time Taken: 0 sec
End of epoch 389 / 700 	 Time Taken: 0 sec
(epoch: 390, iters: 785, time: 0.053) G1_GAN: 1.323 G1_L1: 8.709 G1_MSE_gt: 0.536 G1_MSE: 1.651 D1_real: 0.189 D1_fake: 0.137 G_L1: 3.429 
End of epoch 390 / 700 	 Time Taken: 0 sec
End of epoch 391 / 700 	 Time Taken: 0 sec
End of epoch 392 / 700 	 Time Taken: 0 sec
End of epoch 393 / 700 	 Time Taken: 0 sec
End of epoch 394 / 700 	 Time Taken: 0 sec
(epoch: 395, iters: 795, time: 0.061) G1_GAN: 1.448 G1_L1: 8.472 G1_MSE_gt: 0.636 G1_MSE: 1.413 D1_real: 0.292 D1_fake: 0.293 G_L1: 3.422 
End of epoch 395 / 700 	 Time Taken: 0 sec
End of epoch 396 / 700 	 Time Taken: 0 sec
End of epoch 397 / 700 	 Time Taken: 0 sec
End of epoch 398 / 700 	 Time Taken: 0 sec
End of epoch 399 / 700 	 Time Taken: 0 sec
(epoch: 400, iters: 805, time: 0.054) G1_GAN: 1.278 G1_L1: 7.964 G1_MSE_gt: 0.585 G1_MSE: 1.486 D1_real: 0.177 D1_fake: 0.153 G_L1: 3.467 
saving the model at the end of epoch 400, iters 2800
End of epoch 400 / 700 	 Time Taken: 1 sec
End of epoch 401 / 700 	 Time Taken: 0 sec
update learning rate: 0.000200 -> 0.000199
End of epoch 402 / 700 	 Time Taken: 0 sec
update learning rate: 0.000199 -> 0.000199
End of epoch 403 / 700 	 Time Taken: 0 sec
update learning rate: 0.000199 -> 0.000198
End of epoch 404 / 700 	 Time Taken: 0 sec
update learning rate: 0.000198 -> 0.000197
(epoch: 405, iters: 815, time: 0.063) G1_GAN: 1.448 G1_L1: 7.878 G1_MSE_gt: 0.582 G1_MSE: 1.513 D1_real: 0.187 D1_fake: 0.246 G_L1: 3.370 
End of epoch 405 / 700 	 Time Taken: 0 sec
update learning rate: 0.000197 -> 0.000197
End of epoch 406 / 700 	 Time Taken: 0 sec
update learning rate: 0.000197 -> 0.000196
End of epoch 407 / 700 	 Time Taken: 0 sec
update learning rate: 0.000196 -> 0.000195
End of epoch 408 / 700 	 Time Taken: 0 sec
update learning rate: 0.000195 -> 0.000195
End of epoch 409 / 700 	 Time Taken: 0 sec
update learning rate: 0.000195 -> 0.000194
(epoch: 410, iters: 825, time: 0.054) G1_GAN: 1.312 G1_L1: 7.969 G1_MSE_gt: 0.583 G1_MSE: 1.977 D1_real: 0.125 D1_fake: 0.145 G_L1: 3.459 
End of epoch 410 / 700 	 Time Taken: 0 sec
update learning rate: 0.000194 -> 0.000193
End of epoch 411 / 700 	 Time Taken: 0 sec
update learning rate: 0.000193 -> 0.000193
End of epoch 412 / 700 	 Time Taken: 0 sec
update learning rate: 0.000193 -> 0.000192
End of epoch 413 / 700 	 Time Taken: 0 sec
update learning rate: 0.000192 -> 0.000191
End of epoch 414 / 700 	 Time Taken: 0 sec
update learning rate: 0.000191 -> 0.000191
(epoch: 415, iters: 835, time: 0.063) G1_GAN: 1.403 G1_L1: 10.301 G1_MSE_gt: 3.552 G1_MSE: 1.768 D1_real: 0.125 D1_fake: 0.135 G_L1: 3.339 
End of epoch 415 / 700 	 Time Taken: 0 sec
update learning rate: 0.000191 -> 0.000190
End of epoch 416 / 700 	 Time Taken: 0 sec
update learning rate: 0.000190 -> 0.000189
End of epoch 417 / 700 	 Time Taken: 0 sec
update learning rate: 0.000189 -> 0.000189
End of epoch 418 / 700 	 Time Taken: 0 sec
update learning rate: 0.000189 -> 0.000188
End of epoch 419 / 700 	 Time Taken: 0 sec
update learning rate: 0.000188 -> 0.000187
(epoch: 420, iters: 845, time: 0.055) G1_GAN: 1.547 G1_L1: 7.769 G1_MSE_gt: 0.710 G1_MSE: 1.550 D1_real: 0.155 D1_fake: 0.195 G_L1: 3.701 
End of epoch 420 / 700 	 Time Taken: 0 sec
update learning rate: 0.000187 -> 0.000187
End of epoch 421 / 700 	 Time Taken: 0 sec
update learning rate: 0.000187 -> 0.000186
End of epoch 422 / 700 	 Time Taken: 0 sec
update learning rate: 0.000186 -> 0.000185
End of epoch 423 / 700 	 Time Taken: 0 sec
update learning rate: 0.000185 -> 0.000185
End of epoch 424 / 700 	 Time Taken: 0 sec
update learning rate: 0.000185 -> 0.000184
(epoch: 425, iters: 855, time: 0.062) G1_GAN: 1.411 G1_L1: 7.738 G1_MSE_gt: 0.583 G1_MSE: 1.811 D1_real: 0.098 D1_fake: 0.167 G_L1: 3.597 
End of epoch 425 / 700 	 Time Taken: 0 sec
update learning rate: 0.000184 -> 0.000183
End of epoch 426 / 700 	 Time Taken: 0 sec
update learning rate: 0.000183 -> 0.000183
End of epoch 427 / 700 	 Time Taken: 0 sec
update learning rate: 0.000183 -> 0.000182
End of epoch 428 / 700 	 Time Taken: 0 sec
update learning rate: 0.000182 -> 0.000181
End of epoch 429 / 700 	 Time Taken: 0 sec
update learning rate: 0.000181 -> 0.000181
(epoch: 430, iters: 865, time: 0.055) G1_GAN: 1.420 G1_L1: 7.686 G1_MSE_gt: 0.588 G1_MSE: 1.669 D1_real: 0.184 D1_fake: 0.113 G_L1: 3.488 
End of epoch 430 / 700 	 Time Taken: 0 sec
update learning rate: 0.000181 -> 0.000180
End of epoch 431 / 700 	 Time Taken: 0 sec
update learning rate: 0.000180 -> 0.000179
End of epoch 432 / 700 	 Time Taken: 0 sec
update learning rate: 0.000179 -> 0.000179
End of epoch 433 / 700 	 Time Taken: 0 sec
update learning rate: 0.000179 -> 0.000178
End of epoch 434 / 700 	 Time Taken: 0 sec
update learning rate: 0.000178 -> 0.000177
(epoch: 435, iters: 875, time: 0.064) G1_GAN: 1.515 G1_L1: 7.728 G1_MSE_gt: 0.546 G1_MSE: 2.043 D1_real: 0.128 D1_fake: 0.172 G_L1: 3.447 
End of epoch 435 / 700 	 Time Taken: 0 sec
update learning rate: 0.000177 -> 0.000177
End of epoch 436 / 700 	 Time Taken: 0 sec
update learning rate: 0.000177 -> 0.000176
End of epoch 437 / 700 	 Time Taken: 0 sec
update learning rate: 0.000176 -> 0.000175
End of epoch 438 / 700 	 Time Taken: 0 sec
update learning rate: 0.000175 -> 0.000175
End of epoch 439 / 700 	 Time Taken: 0 sec
update learning rate: 0.000175 -> 0.000174
(epoch: 440, iters: 885, time: 0.054) G1_GAN: 1.398 G1_L1: 7.251 G1_MSE_gt: 0.562 G1_MSE: 1.623 D1_real: 0.179 D1_fake: 0.134 G_L1: 3.411 
End of epoch 440 / 700 	 Time Taken: 0 sec
update learning rate: 0.000174 -> 0.000173
End of epoch 441 / 700 	 Time Taken: 0 sec
update learning rate: 0.000173 -> 0.000173
End of epoch 442 / 700 	 Time Taken: 0 sec
update learning rate: 0.000173 -> 0.000172
End of epoch 443 / 700 	 Time Taken: 0 sec
update learning rate: 0.000172 -> 0.000171
End of epoch 444 / 700 	 Time Taken: 0 sec
update learning rate: 0.000171 -> 0.000171
(epoch: 445, iters: 895, time: 0.064) G1_GAN: 1.261 G1_L1: 6.983 G1_MSE_gt: 0.549 G1_MSE: 1.470 D1_real: 0.116 D1_fake: 0.189 G_L1: 3.390 
End of epoch 445 / 700 	 Time Taken: 0 sec
update learning rate: 0.000171 -> 0.000170
End of epoch 446 / 700 	 Time Taken: 0 sec
update learning rate: 0.000170 -> 0.000169
End of epoch 447 / 700 	 Time Taken: 0 sec
update learning rate: 0.000169 -> 0.000169
End of epoch 448 / 700 	 Time Taken: 0 sec
update learning rate: 0.000169 -> 0.000168
End of epoch 449 / 700 	 Time Taken: 0 sec
update learning rate: 0.000168 -> 0.000167
(epoch: 450, iters: 905, time: 0.062) G1_GAN: 1.224 G1_L1: 6.767 G1_MSE_gt: 0.554 G1_MSE: 1.451 D1_real: 0.103 D1_fake: 0.125 G_L1: 3.401 
End of epoch 450 / 700 	 Time Taken: 0 sec
update learning rate: 0.000167 -> 0.000167
End of epoch 451 / 700 	 Time Taken: 0 sec
update learning rate: 0.000167 -> 0.000166
End of epoch 452 / 700 	 Time Taken: 0 sec
update learning rate: 0.000166 -> 0.000165
End of epoch 453 / 700 	 Time Taken: 0 sec
update learning rate: 0.000165 -> 0.000165
End of epoch 454 / 700 	 Time Taken: 0 sec
update learning rate: 0.000165 -> 0.000164
(epoch: 455, iters: 915, time: 0.055) G1_GAN: 1.401 G1_L1: 6.876 G1_MSE_gt: 0.536 G1_MSE: 1.510 D1_real: 0.137 D1_fake: 0.132 G_L1: 3.366 
End of epoch 455 / 700 	 Time Taken: 0 sec
update learning rate: 0.000164 -> 0.000163
End of epoch 456 / 700 	 Time Taken: 0 sec
update learning rate: 0.000163 -> 0.000163
End of epoch 457 / 700 	 Time Taken: 0 sec
update learning rate: 0.000163 -> 0.000162
End of epoch 458 / 700 	 Time Taken: 0 sec
update learning rate: 0.000162 -> 0.000161
End of epoch 459 / 700 	 Time Taken: 0 sec
update learning rate: 0.000161 -> 0.000161
(epoch: 460, iters: 925, time: 0.063) G1_GAN: 1.491 G1_L1: 6.921 G1_MSE_gt: 0.580 G1_MSE: 1.663 D1_real: 0.097 D1_fake: 0.228 G_L1: 3.322 
End of epoch 460 / 700 	 Time Taken: 0 sec
update learning rate: 0.000161 -> 0.000160
End of epoch 461 / 700 	 Time Taken: 0 sec
update learning rate: 0.000160 -> 0.000159
End of epoch 462 / 700 	 Time Taken: 0 sec
update learning rate: 0.000159 -> 0.000159
End of epoch 463 / 700 	 Time Taken: 0 sec
update learning rate: 0.000159 -> 0.000158
End of epoch 464 / 700 	 Time Taken: 0 sec
update learning rate: 0.000158 -> 0.000157
(epoch: 465, iters: 935, time: 0.056) G1_GAN: 1.292 G1_L1: 7.086 G1_MSE_gt: 0.559 G1_MSE: 1.464 D1_real: 0.129 D1_fake: 0.145 G_L1: 3.291 
End of epoch 465 / 700 	 Time Taken: 0 sec
update learning rate: 0.000157 -> 0.000157
End of epoch 466 / 700 	 Time Taken: 0 sec
update learning rate: 0.000157 -> 0.000156
End of epoch 467 / 700 	 Time Taken: 0 sec
update learning rate: 0.000156 -> 0.000155
End of epoch 468 / 700 	 Time Taken: 0 sec
update learning rate: 0.000155 -> 0.000155
End of epoch 469 / 700 	 Time Taken: 0 sec
update learning rate: 0.000155 -> 0.000154
(epoch: 470, iters: 945, time: 0.065) G1_GAN: 1.513 G1_L1: 6.769 G1_MSE_gt: 0.575 G1_MSE: 1.586 D1_real: 0.149 D1_fake: 0.111 G_L1: 3.291 
End of epoch 470 / 700 	 Time Taken: 0 sec
update learning rate: 0.000154 -> 0.000153
End of epoch 471 / 700 	 Time Taken: 0 sec
update learning rate: 0.000153 -> 0.000153
End of epoch 472 / 700 	 Time Taken: 0 sec
update learning rate: 0.000153 -> 0.000152
End of epoch 473 / 700 	 Time Taken: 0 sec
update learning rate: 0.000152 -> 0.000151
End of epoch 474 / 700 	 Time Taken: 0 sec
update learning rate: 0.000151 -> 0.000151
(epoch: 475, iters: 955, time: 0.063) G1_GAN: 1.255 G1_L1: 6.772 G1_MSE_gt: 0.535 G1_MSE: 1.541 D1_real: 0.129 D1_fake: 0.123 G_L1: 3.259 
End of epoch 475 / 700 	 Time Taken: 0 sec
update learning rate: 0.000151 -> 0.000150
End of epoch 476 / 700 	 Time Taken: 0 sec
update learning rate: 0.000150 -> 0.000149
End of epoch 477 / 700 	 Time Taken: 0 sec
update learning rate: 0.000149 -> 0.000149
End of epoch 478 / 700 	 Time Taken: 0 sec
update learning rate: 0.000149 -> 0.000148
End of epoch 479 / 700 	 Time Taken: 0 sec
update learning rate: 0.000148 -> 0.000147
(epoch: 480, iters: 965, time: 0.057) G1_GAN: 1.596 G1_L1: 6.710 G1_MSE_gt: 0.541 G1_MSE: 1.588 D1_real: 0.130 D1_fake: 0.161 G_L1: 3.235 
End of epoch 480 / 700 	 Time Taken: 0 sec
update learning rate: 0.000147 -> 0.000147
End of epoch 481 / 700 	 Time Taken: 0 sec
update learning rate: 0.000147 -> 0.000146
End of epoch 482 / 700 	 Time Taken: 0 sec
update learning rate: 0.000146 -> 0.000145
End of epoch 483 / 700 	 Time Taken: 0 sec
update learning rate: 0.000145 -> 0.000145
End of epoch 484 / 700 	 Time Taken: 0 sec
update learning rate: 0.000145 -> 0.000144
(epoch: 485, iters: 975, time: 0.065) G1_GAN: 1.629 G1_L1: 6.629 G1_MSE_gt: 0.522 G1_MSE: 2.112 D1_real: 0.138 D1_fake: 0.146 G_L1: 3.259 
End of epoch 485 / 700 	 Time Taken: 0 sec
update learning rate: 0.000144 -> 0.000143
End of epoch 486 / 700 	 Time Taken: 0 sec
update learning rate: 0.000143 -> 0.000143
End of epoch 487 / 700 	 Time Taken: 0 sec
update learning rate: 0.000143 -> 0.000142
End of epoch 488 / 700 	 Time Taken: 0 sec
update learning rate: 0.000142 -> 0.000141
End of epoch 489 / 700 	 Time Taken: 0 sec
update learning rate: 0.000141 -> 0.000141
(epoch: 490, iters: 985, time: 0.056) G1_GAN: 1.580 G1_L1: 6.756 G1_MSE_gt: 0.556 G1_MSE: 1.797 D1_real: 0.107 D1_fake: 0.228 G_L1: 3.230 
End of epoch 490 / 700 	 Time Taken: 0 sec
update learning rate: 0.000141 -> 0.000140
End of epoch 491 / 700 	 Time Taken: 0 sec
update learning rate: 0.000140 -> 0.000139
End of epoch 492 / 700 	 Time Taken: 0 sec
update learning rate: 0.000139 -> 0.000139
End of epoch 493 / 700 	 Time Taken: 0 sec
update learning rate: 0.000139 -> 0.000138
End of epoch 494 / 700 	 Time Taken: 0 sec
update learning rate: 0.000138 -> 0.000137
(epoch: 495, iters: 995, time: 0.057) G1_GAN: 1.572 G1_L1: 6.504 G1_MSE_gt: 0.527 G1_MSE: 1.560 D1_real: 0.121 D1_fake: 0.138 G_L1: 3.199 
End of epoch 495 / 700 	 Time Taken: 0 sec
update learning rate: 0.000137 -> 0.000137
End of epoch 496 / 700 	 Time Taken: 0 sec
update learning rate: 0.000137 -> 0.000136
End of epoch 497 / 700 	 Time Taken: 0 sec
update learning rate: 0.000136 -> 0.000135
End of epoch 498 / 700 	 Time Taken: 0 sec
update learning rate: 0.000135 -> 0.000135
End of epoch 499 / 700 	 Time Taken: 0 sec
update learning rate: 0.000135 -> 0.000134
(epoch: 500, iters: 1005, time: 0.064) G1_GAN: 1.482 G1_L1: 6.586 G1_MSE_gt: 0.523 G1_MSE: 1.404 D1_real: 0.143 D1_fake: 0.124 G_L1: 3.213 
saving the model at the end of epoch 500, iters 3500
End of epoch 500 / 700 	 Time Taken: 1 sec
update learning rate: 0.000134 -> 0.000133
End of epoch 501 / 700 	 Time Taken: 0 sec
update learning rate: 0.000133 -> 0.000133
End of epoch 502 / 700 	 Time Taken: 0 sec
update learning rate: 0.000133 -> 0.000132
End of epoch 503 / 700 	 Time Taken: 0 sec
update learning rate: 0.000132 -> 0.000131
End of epoch 504 / 700 	 Time Taken: 0 sec
update learning rate: 0.000131 -> 0.000131
(epoch: 505, iters: 1015, time: 0.057) G1_GAN: 1.343 G1_L1: 6.403 G1_MSE_gt: 0.542 G1_MSE: 1.804 D1_real: 0.090 D1_fake: 0.172 G_L1: 3.190 
End of epoch 505 / 700 	 Time Taken: 0 sec
update learning rate: 0.000131 -> 0.000130
End of epoch 506 / 700 	 Time Taken: 0 sec
update learning rate: 0.000130 -> 0.000129
End of epoch 507 / 700 	 Time Taken: 0 sec
update learning rate: 0.000129 -> 0.000129
End of epoch 508 / 700 	 Time Taken: 0 sec
update learning rate: 0.000129 -> 0.000128
End of epoch 509 / 700 	 Time Taken: 0 sec
update learning rate: 0.000128 -> 0.000127
(epoch: 510, iters: 1025, time: 0.066) G1_GAN: 1.289 G1_L1: 6.477 G1_MSE_gt: 0.561 G1_MSE: 1.503 D1_real: 0.121 D1_fake: 0.142 G_L1: 3.151 
End of epoch 510 / 700 	 Time Taken: 0 sec
update learning rate: 0.000127 -> 0.000127
End of epoch 511 / 700 	 Time Taken: 0 sec
update learning rate: 0.000127 -> 0.000126
End of epoch 512 / 700 	 Time Taken: 0 sec
update learning rate: 0.000126 -> 0.000125
End of epoch 513 / 700 	 Time Taken: 0 sec
update learning rate: 0.000125 -> 0.000125
End of epoch 514 / 700 	 Time Taken: 0 sec
update learning rate: 0.000125 -> 0.000124
(epoch: 515, iters: 1035, time: 0.065) G1_GAN: 1.590 G1_L1: 6.296 G1_MSE_gt: 0.548 G1_MSE: 1.837 D1_real: 0.088 D1_fake: 0.136 G_L1: 3.136 
End of epoch 515 / 700 	 Time Taken: 0 sec
update learning rate: 0.000124 -> 0.000123
End of epoch 516 / 700 	 Time Taken: 0 sec
update learning rate: 0.000123 -> 0.000123
End of epoch 517 / 700 	 Time Taken: 0 sec
update learning rate: 0.000123 -> 0.000122
End of epoch 518 / 700 	 Time Taken: 0 sec
update learning rate: 0.000122 -> 0.000121
End of epoch 519 / 700 	 Time Taken: 0 sec
update learning rate: 0.000121 -> 0.000121
(epoch: 520, iters: 1045, time: 0.058) G1_GAN: 1.330 G1_L1: 6.260 G1_MSE_gt: 0.554 G1_MSE: 1.291 D1_real: 0.129 D1_fake: 0.094 G_L1: 3.144 
End of epoch 520 / 700 	 Time Taken: 0 sec
update learning rate: 0.000121 -> 0.000120
End of epoch 521 / 700 	 Time Taken: 0 sec
update learning rate: 0.000120 -> 0.000119
End of epoch 522 / 700 	 Time Taken: 0 sec
update learning rate: 0.000119 -> 0.000119
End of epoch 523 / 700 	 Time Taken: 0 sec
update learning rate: 0.000119 -> 0.000118
End of epoch 524 / 700 	 Time Taken: 0 sec
update learning rate: 0.000118 -> 0.000117
(epoch: 525, iters: 1055, time: 0.068) G1_GAN: 1.313 G1_L1: 6.274 G1_MSE_gt: 0.539 G1_MSE: 1.852 D1_real: 0.096 D1_fake: 0.115 G_L1: 3.145 
End of epoch 525 / 700 	 Time Taken: 0 sec
update learning rate: 0.000117 -> 0.000117
End of epoch 526 / 700 	 Time Taken: 0 sec
update learning rate: 0.000117 -> 0.000116
End of epoch 527 / 700 	 Time Taken: 0 sec
update learning rate: 0.000116 -> 0.000115
End of epoch 528 / 700 	 Time Taken: 0 sec
update learning rate: 0.000115 -> 0.000115
End of epoch 529 / 700 	 Time Taken: 0 sec
update learning rate: 0.000115 -> 0.000114
(epoch: 530, iters: 1065, time: 0.064) G1_GAN: 1.440 G1_L1: 6.114 G1_MSE_gt: 0.525 G1_MSE: 1.855 D1_real: 0.103 D1_fake: 0.137 G_L1: 3.142 
End of epoch 530 / 700 	 Time Taken: 0 sec
update learning rate: 0.000114 -> 0.000113
End of epoch 531 / 700 	 Time Taken: 0 sec
update learning rate: 0.000113 -> 0.000113
End of epoch 532 / 700 	 Time Taken: 0 sec
update learning rate: 0.000113 -> 0.000112
End of epoch 533 / 700 	 Time Taken: 0 sec
update learning rate: 0.000112 -> 0.000111
End of epoch 534 / 700 	 Time Taken: 0 sec
update learning rate: 0.000111 -> 0.000111
(epoch: 535, iters: 1075, time: 0.066) G1_GAN: 1.394 G1_L1: 6.251 G1_MSE_gt: 0.538 G1_MSE: 1.593 D1_real: 0.109 D1_fake: 0.120 G_L1: 3.116 
End of epoch 535 / 700 	 Time Taken: 0 sec
update learning rate: 0.000111 -> 0.000110
End of epoch 536 / 700 	 Time Taken: 0 sec
update learning rate: 0.000110 -> 0.000109
End of epoch 537 / 700 	 Time Taken: 0 sec
update learning rate: 0.000109 -> 0.000109
End of epoch 538 / 700 	 Time Taken: 0 sec
update learning rate: 0.000109 -> 0.000108
End of epoch 539 / 700 	 Time Taken: 0 sec
update learning rate: 0.000108 -> 0.000107
(epoch: 540, iters: 1085, time: 0.062) G1_GAN: 1.388 G1_L1: 6.063 G1_MSE_gt: 0.518 G1_MSE: 1.423 D1_real: 0.103 D1_fake: 0.097 G_L1: 3.111 
End of epoch 540 / 700 	 Time Taken: 0 sec
update learning rate: 0.000107 -> 0.000107
End of epoch 541 / 700 	 Time Taken: 0 sec
update learning rate: 0.000107 -> 0.000106
End of epoch 542 / 700 	 Time Taken: 0 sec
update learning rate: 0.000106 -> 0.000105
End of epoch 543 / 700 	 Time Taken: 0 sec
update learning rate: 0.000105 -> 0.000105
End of epoch 544 / 700 	 Time Taken: 0 sec
update learning rate: 0.000105 -> 0.000104
(epoch: 545, iters: 1095, time: 0.067) G1_GAN: 1.266 G1_L1: 6.177 G1_MSE_gt: 0.507 G1_MSE: 1.377 D1_real: 0.102 D1_fake: 0.105 G_L1: 3.121 
End of epoch 545 / 700 	 Time Taken: 0 sec
update learning rate: 0.000104 -> 0.000103
End of epoch 546 / 700 	 Time Taken: 0 sec
update learning rate: 0.000103 -> 0.000103
End of epoch 547 / 700 	 Time Taken: 0 sec
update learning rate: 0.000103 -> 0.000102
End of epoch 548 / 700 	 Time Taken: 0 sec
update learning rate: 0.000102 -> 0.000101
End of epoch 549 / 700 	 Time Taken: 0 sec
update learning rate: 0.000101 -> 0.000101
(epoch: 550, iters: 1105, time: 0.066) G1_GAN: 1.385 G1_L1: 6.148 G1_MSE_gt: 0.529 G1_MSE: 1.943 D1_real: 0.088 D1_fake: 0.137 G_L1: 3.086 
End of epoch 550 / 700 	 Time Taken: 0 sec
update learning rate: 0.000101 -> 0.000100
End of epoch 551 / 700 	 Time Taken: 0 sec
update learning rate: 0.000100 -> 0.000099
End of epoch 552 / 700 	 Time Taken: 0 sec
update learning rate: 0.000099 -> 0.000099
End of epoch 553 / 700 	 Time Taken: 0 sec
update learning rate: 0.000099 -> 0.000098
End of epoch 554 / 700 	 Time Taken: 0 sec
update learning rate: 0.000098 -> 0.000097
(epoch: 555, iters: 1115, time: 0.058) G1_GAN: 1.459 G1_L1: 6.046 G1_MSE_gt: 0.544 G1_MSE: 2.476 D1_real: 0.086 D1_fake: 0.121 G_L1: 3.044 
End of epoch 555 / 700 	 Time Taken: 0 sec
update learning rate: 0.000097 -> 0.000097
End of epoch 556 / 700 	 Time Taken: 0 sec
update learning rate: 0.000097 -> 0.000096
End of epoch 557 / 700 	 Time Taken: 0 sec
update learning rate: 0.000096 -> 0.000095
End of epoch 558 / 700 	 Time Taken: 0 sec
update learning rate: 0.000095 -> 0.000095
End of epoch 559 / 700 	 Time Taken: 0 sec
update learning rate: 0.000095 -> 0.000094
(epoch: 560, iters: 1125, time: 0.068) G1_GAN: 1.593 G1_L1: 6.051 G1_MSE_gt: 0.535 G1_MSE: 1.791 D1_real: 0.111 D1_fake: 0.119 G_L1: 3.044 
End of epoch 560 / 700 	 Time Taken: 0 sec
update learning rate: 0.000094 -> 0.000093
End of epoch 561 / 700 	 Time Taken: 0 sec
update learning rate: 0.000093 -> 0.000093
End of epoch 562 / 700 	 Time Taken: 0 sec
update learning rate: 0.000093 -> 0.000092
End of epoch 563 / 700 	 Time Taken: 0 sec
update learning rate: 0.000092 -> 0.000091
End of epoch 564 / 700 	 Time Taken: 0 sec
update learning rate: 0.000091 -> 0.000091
(epoch: 565, iters: 1135, time: 0.067) G1_GAN: 1.602 G1_L1: 6.042 G1_MSE_gt: 0.564 G1_MSE: 1.668 D1_real: 0.104 D1_fake: 0.117 G_L1: 3.032 
End of epoch 565 / 700 	 Time Taken: 0 sec
update learning rate: 0.000091 -> 0.000090
End of epoch 566 / 700 	 Time Taken: 0 sec
update learning rate: 0.000090 -> 0.000089
End of epoch 567 / 700 	 Time Taken: 0 sec
update learning rate: 0.000089 -> 0.000089
End of epoch 568 / 700 	 Time Taken: 0 sec
update learning rate: 0.000089 -> 0.000088
End of epoch 569 / 700 	 Time Taken: 0 sec
update learning rate: 0.000088 -> 0.000087
(epoch: 570, iters: 1145, time: 0.067) G1_GAN: 1.457 G1_L1: 5.962 G1_MSE_gt: 0.501 G1_MSE: 1.796 D1_real: 0.094 D1_fake: 0.144 G_L1: 3.043 
End of epoch 570 / 700 	 Time Taken: 0 sec
update learning rate: 0.000087 -> 0.000087
End of epoch 571 / 700 	 Time Taken: 0 sec
update learning rate: 0.000087 -> 0.000086
End of epoch 572 / 700 	 Time Taken: 0 sec
update learning rate: 0.000086 -> 0.000085
End of epoch 573 / 700 	 Time Taken: 0 sec
update learning rate: 0.000085 -> 0.000085
End of epoch 574 / 700 	 Time Taken: 0 sec
update learning rate: 0.000085 -> 0.000084
(epoch: 575, iters: 1155, time: 0.060) G1_GAN: 1.462 G1_L1: 5.868 G1_MSE_gt: 0.516 G1_MSE: 1.621 D1_real: 0.098 D1_fake: 0.127 G_L1: 3.027 
End of epoch 575 / 700 	 Time Taken: 0 sec
update learning rate: 0.000084 -> 0.000083
End of epoch 576 / 700 	 Time Taken: 0 sec
update learning rate: 0.000083 -> 0.000083
End of epoch 577 / 700 	 Time Taken: 0 sec
update learning rate: 0.000083 -> 0.000082
End of epoch 578 / 700 	 Time Taken: 0 sec
update learning rate: 0.000082 -> 0.000081
End of epoch 579 / 700 	 Time Taken: 0 sec
update learning rate: 0.000081 -> 0.000081
(epoch: 580, iters: 1165, time: 0.071) G1_GAN: 1.362 G1_L1: 5.959 G1_MSE_gt: 0.516 G1_MSE: 1.349 D1_real: 0.100 D1_fake: 0.133 G_L1: 3.028 
End of epoch 580 / 700 	 Time Taken: 0 sec
update learning rate: 0.000081 -> 0.000080
End of epoch 581 / 700 	 Time Taken: 0 sec
update learning rate: 0.000080 -> 0.000079
End of epoch 582 / 700 	 Time Taken: 0 sec
update learning rate: 0.000079 -> 0.000079
End of epoch 583 / 700 	 Time Taken: 0 sec
update learning rate: 0.000079 -> 0.000078
End of epoch 584 / 700 	 Time Taken: 0 sec
update learning rate: 0.000078 -> 0.000077
(epoch: 585, iters: 1175, time: 0.067) G1_GAN: 1.496 G1_L1: 5.895 G1_MSE_gt: 0.495 G1_MSE: 1.996 D1_real: 0.102 D1_fake: 0.091 G_L1: 2.990 
End of epoch 585 / 700 	 Time Taken: 0 sec
update learning rate: 0.000077 -> 0.000077
End of epoch 586 / 700 	 Time Taken: 0 sec
update learning rate: 0.000077 -> 0.000076
End of epoch 587 / 700 	 Time Taken: 0 sec
update learning rate: 0.000076 -> 0.000075
End of epoch 588 / 700 	 Time Taken: 0 sec
update learning rate: 0.000075 -> 0.000075
End of epoch 589 / 700 	 Time Taken: 0 sec
update learning rate: 0.000075 -> 0.000074
(epoch: 590, iters: 1185, time: 0.068) G1_GAN: 1.538 G1_L1: 5.873 G1_MSE_gt: 0.537 G1_MSE: 1.482 D1_real: 0.118 D1_fake: 0.121 G_L1: 2.986 
End of epoch 590 / 700 	 Time Taken: 0 sec
update learning rate: 0.000074 -> 0.000073
End of epoch 591 / 700 	 Time Taken: 0 sec
update learning rate: 0.000073 -> 0.000073
End of epoch 592 / 700 	 Time Taken: 0 sec
update learning rate: 0.000073 -> 0.000072
End of epoch 593 / 700 	 Time Taken: 0 sec
update learning rate: 0.000072 -> 0.000071
End of epoch 594 / 700 	 Time Taken: 0 sec
update learning rate: 0.000071 -> 0.000071
(epoch: 595, iters: 1195, time: 0.060) G1_GAN: 1.518 G1_L1: 5.832 G1_MSE_gt: 0.503 G1_MSE: 1.551 D1_real: 0.093 D1_fake: 0.135 G_L1: 3.007 
End of epoch 595 / 700 	 Time Taken: 0 sec
update learning rate: 0.000071 -> 0.000070
End of epoch 596 / 700 	 Time Taken: 0 sec
update learning rate: 0.000070 -> 0.000069
End of epoch 597 / 700 	 Time Taken: 0 sec
update learning rate: 0.000069 -> 0.000069
End of epoch 598 / 700 	 Time Taken: 0 sec
update learning rate: 0.000069 -> 0.000068
End of epoch 599 / 700 	 Time Taken: 0 sec
update learning rate: 0.000068 -> 0.000067
(epoch: 600, iters: 1205, time: 0.070) G1_GAN: 1.407 G1_L1: 5.858 G1_MSE_gt: 0.515 G1_MSE: 1.475 D1_real: 0.084 D1_fake: 0.184 G_L1: 2.965 
saving the model at the end of epoch 600, iters 4200
End of epoch 600 / 700 	 Time Taken: 1 sec
update learning rate: 0.000067 -> 0.000067
End of epoch 601 / 700 	 Time Taken: 0 sec
update learning rate: 0.000067 -> 0.000066
End of epoch 602 / 700 	 Time Taken: 0 sec
update learning rate: 0.000066 -> 0.000065
End of epoch 603 / 700 	 Time Taken: 0 sec
update learning rate: 0.000065 -> 0.000065
End of epoch 604 / 700 	 Time Taken: 0 sec
update learning rate: 0.000065 -> 0.000064
(epoch: 605, iters: 1215, time: 0.067) G1_GAN: 1.515 G1_L1: 5.846 G1_MSE_gt: 0.506 G1_MSE: 1.950 D1_real: 0.085 D1_fake: 0.090 G_L1: 3.000 
End of epoch 605 / 700 	 Time Taken: 0 sec
update learning rate: 0.000064 -> 0.000063
End of epoch 606 / 700 	 Time Taken: 0 sec
update learning rate: 0.000063 -> 0.000063
End of epoch 607 / 700 	 Time Taken: 0 sec
update learning rate: 0.000063 -> 0.000062
End of epoch 608 / 700 	 Time Taken: 0 sec
update learning rate: 0.000062 -> 0.000061
End of epoch 609 / 700 	 Time Taken: 0 sec
update learning rate: 0.000061 -> 0.000061
(epoch: 610, iters: 1225, time: 0.068) G1_GAN: 1.373 G1_L1: 5.853 G1_MSE_gt: 0.517 G1_MSE: 1.377 D1_real: 0.098 D1_fake: 0.106 G_L1: 2.967 
End of epoch 610 / 700 	 Time Taken: 0 sec
update learning rate: 0.000061 -> 0.000060
End of epoch 611 / 700 	 Time Taken: 0 sec
update learning rate: 0.000060 -> 0.000059
End of epoch 612 / 700 	 Time Taken: 0 sec
update learning rate: 0.000059 -> 0.000059
End of epoch 613 / 700 	 Time Taken: 0 sec
update learning rate: 0.000059 -> 0.000058
End of epoch 614 / 700 	 Time Taken: 0 sec
update learning rate: 0.000058 -> 0.000057
(epoch: 615, iters: 1235, time: 0.060) G1_GAN: 1.465 G1_L1: 5.798 G1_MSE_gt: 0.528 G1_MSE: 1.704 D1_real: 0.108 D1_fake: 0.103 G_L1: 2.961 
End of epoch 615 / 700 	 Time Taken: 0 sec
update learning rate: 0.000057 -> 0.000057
End of epoch 616 / 700 	 Time Taken: 0 sec
update learning rate: 0.000057 -> 0.000056
End of epoch 617 / 700 	 Time Taken: 0 sec
update learning rate: 0.000056 -> 0.000055
End of epoch 618 / 700 	 Time Taken: 0 sec
update learning rate: 0.000055 -> 0.000055
End of epoch 619 / 700 	 Time Taken: 0 sec
update learning rate: 0.000055 -> 0.000054
(epoch: 620, iters: 1245, time: 0.070) G1_GAN: 1.480 G1_L1: 5.868 G1_MSE_gt: 0.498 G1_MSE: 1.752 D1_real: 0.089 D1_fake: 0.107 G_L1: 2.948 
End of epoch 620 / 700 	 Time Taken: 0 sec
update learning rate: 0.000054 -> 0.000053
End of epoch 621 / 700 	 Time Taken: 0 sec
update learning rate: 0.000053 -> 0.000053
End of epoch 622 / 700 	 Time Taken: 0 sec
update learning rate: 0.000053 -> 0.000052
End of epoch 623 / 700 	 Time Taken: 0 sec
update learning rate: 0.000052 -> 0.000051
End of epoch 624 / 700 	 Time Taken: 0 sec
update learning rate: 0.000051 -> 0.000051
(epoch: 625, iters: 1255, time: 0.068) G1_GAN: 1.540 G1_L1: 5.752 G1_MSE_gt: 0.511 G1_MSE: 1.427 D1_real: 0.100 D1_fake: 0.186 G_L1: 2.935 
End of epoch 625 / 700 	 Time Taken: 0 sec
update learning rate: 0.000051 -> 0.000050
End of epoch 626 / 700 	 Time Taken: 0 sec
update learning rate: 0.000050 -> 0.000049
End of epoch 627 / 700 	 Time Taken: 0 sec
update learning rate: 0.000049 -> 0.000049
End of epoch 628 / 700 	 Time Taken: 0 sec
update learning rate: 0.000049 -> 0.000048
End of epoch 629 / 700 	 Time Taken: 0 sec
update learning rate: 0.000048 -> 0.000047
(epoch: 630, iters: 1265, time: 0.069) G1_GAN: 1.569 G1_L1: 5.804 G1_MSE_gt: 0.537 G1_MSE: 1.845 D1_real: 0.090 D1_fake: 0.102 G_L1: 2.949 
End of epoch 630 / 700 	 Time Taken: 0 sec
update learning rate: 0.000047 -> 0.000047
End of epoch 631 / 700 	 Time Taken: 0 sec
update learning rate: 0.000047 -> 0.000046
End of epoch 632 / 700 	 Time Taken: 0 sec
update learning rate: 0.000046 -> 0.000045
End of epoch 633 / 700 	 Time Taken: 0 sec
update learning rate: 0.000045 -> 0.000045
End of epoch 634 / 700 	 Time Taken: 0 sec
update learning rate: 0.000045 -> 0.000044
(epoch: 635, iters: 1275, time: 0.069) G1_GAN: 1.537 G1_L1: 5.612 G1_MSE_gt: 0.533 G1_MSE: 1.760 D1_real: 0.106 D1_fake: 0.135 G_L1: 2.904 
End of epoch 635 / 700 	 Time Taken: 0 sec
update learning rate: 0.000044 -> 0.000043
End of epoch 636 / 700 	 Time Taken: 0 sec
update learning rate: 0.000043 -> 0.000043
End of epoch 637 / 700 	 Time Taken: 0 sec
update learning rate: 0.000043 -> 0.000042
End of epoch 638 / 700 	 Time Taken: 0 sec
update learning rate: 0.000042 -> 0.000041
End of epoch 639 / 700 	 Time Taken: 0 sec
update learning rate: 0.000041 -> 0.000041
(epoch: 640, iters: 1285, time: 0.069) G1_GAN: 1.581 G1_L1: 5.759 G1_MSE_gt: 0.494 G1_MSE: 1.700 D1_real: 0.088 D1_fake: 0.136 G_L1: 2.950 
End of epoch 640 / 700 	 Time Taken: 0 sec
update learning rate: 0.000041 -> 0.000040
End of epoch 641 / 700 	 Time Taken: 0 sec
update learning rate: 0.000040 -> 0.000039
End of epoch 642 / 700 	 Time Taken: 0 sec
update learning rate: 0.000039 -> 0.000039
End of epoch 643 / 700 	 Time Taken: 0 sec
update learning rate: 0.000039 -> 0.000038
End of epoch 644 / 700 	 Time Taken: 0 sec
update learning rate: 0.000038 -> 0.000037
(epoch: 645, iters: 1295, time: 0.061) G1_GAN: 1.567 G1_L1: 5.697 G1_MSE_gt: 0.510 G1_MSE: 1.628 D1_real: 0.095 D1_fake: 0.162 G_L1: 2.912 
End of epoch 645 / 700 	 Time Taken: 0 sec
update learning rate: 0.000037 -> 0.000037
End of epoch 646 / 700 	 Time Taken: 0 sec
update learning rate: 0.000037 -> 0.000036
End of epoch 647 / 700 	 Time Taken: 0 sec
update learning rate: 0.000036 -> 0.000035
End of epoch 648 / 700 	 Time Taken: 0 sec
update learning rate: 0.000035 -> 0.000035
End of epoch 649 / 700 	 Time Taken: 0 sec
update learning rate: 0.000035 -> 0.000034
(epoch: 650, iters: 1305, time: 0.071) G1_GAN: 1.552 G1_L1: 5.538 G1_MSE_gt: 0.538 G1_MSE: 1.805 D1_real: 0.099 D1_fake: 0.070 G_L1: 2.923 
End of epoch 650 / 700 	 Time Taken: 0 sec
update learning rate: 0.000034 -> 0.000033
End of epoch 651 / 700 	 Time Taken: 0 sec
update learning rate: 0.000033 -> 0.000033
End of epoch 652 / 700 	 Time Taken: 0 sec
update learning rate: 0.000033 -> 0.000032
End of epoch 653 / 700 	 Time Taken: 0 sec
update learning rate: 0.000032 -> 0.000031
End of epoch 654 / 700 	 Time Taken: 0 sec
update learning rate: 0.000031 -> 0.000031
(epoch: 655, iters: 1315, time: 0.068) G1_GAN: 1.530 G1_L1: 5.567 G1_MSE_gt: 0.494 G1_MSE: 1.605 D1_real: 0.097 D1_fake: 0.106 G_L1: 2.914 
End of epoch 655 / 700 	 Time Taken: 0 sec
update learning rate: 0.000031 -> 0.000030
End of epoch 656 / 700 	 Time Taken: 0 sec
update learning rate: 0.000030 -> 0.000029
End of epoch 657 / 700 	 Time Taken: 0 sec
update learning rate: 0.000029 -> 0.000029
End of epoch 658 / 700 	 Time Taken: 0 sec
update learning rate: 0.000029 -> 0.000028
End of epoch 659 / 700 	 Time Taken: 0 sec
update learning rate: 0.000028 -> 0.000027
(epoch: 660, iters: 1325, time: 0.070) G1_GAN: 1.427 G1_L1: 5.650 G1_MSE_gt: 0.493 G1_MSE: 1.672 D1_real: 0.082 D1_fake: 0.196 G_L1: 2.909 
End of epoch 660 / 700 	 Time Taken: 0 sec
update learning rate: 0.000027 -> 0.000027
End of epoch 661 / 700 	 Time Taken: 0 sec
update learning rate: 0.000027 -> 0.000026
End of epoch 662 / 700 	 Time Taken: 0 sec
update learning rate: 0.000026 -> 0.000025
End of epoch 663 / 700 	 Time Taken: 0 sec
update learning rate: 0.000025 -> 0.000025
End of epoch 664 / 700 	 Time Taken: 0 sec
update learning rate: 0.000025 -> 0.000024
(epoch: 665, iters: 1335, time: 0.070) G1_GAN: 1.490 G1_L1: 5.654 G1_MSE_gt: 0.498 G1_MSE: 1.391 D1_real: 0.101 D1_fake: 0.173 G_L1: 2.899 
End of epoch 665 / 700 	 Time Taken: 0 sec
update learning rate: 0.000024 -> 0.000023
End of epoch 666 / 700 	 Time Taken: 0 sec
update learning rate: 0.000023 -> 0.000023
End of epoch 667 / 700 	 Time Taken: 0 sec
update learning rate: 0.000023 -> 0.000022
End of epoch 668 / 700 	 Time Taken: 0 sec
update learning rate: 0.000022 -> 0.000021
End of epoch 669 / 700 	 Time Taken: 0 sec
update learning rate: 0.000021 -> 0.000021
(epoch: 670, iters: 1345, time: 0.070) G1_GAN: 1.663 G1_L1: 5.636 G1_MSE_gt: 0.532 G1_MSE: 1.778 D1_real: 0.085 D1_fake: 0.142 G_L1: 2.902 
End of epoch 670 / 700 	 Time Taken: 0 sec
update learning rate: 0.000021 -> 0.000020
End of epoch 671 / 700 	 Time Taken: 0 sec
update learning rate: 0.000020 -> 0.000019
End of epoch 672 / 700 	 Time Taken: 0 sec
update learning rate: 0.000019 -> 0.000019
End of epoch 673 / 700 	 Time Taken: 0 sec
update learning rate: 0.000019 -> 0.000018
End of epoch 674 / 700 	 Time Taken: 0 sec
update learning rate: 0.000018 -> 0.000017
(epoch: 675, iters: 1355, time: 0.072) G1_GAN: 1.530 G1_L1: 5.599 G1_MSE_gt: 0.487 G1_MSE: 1.586 D1_real: 0.101 D1_fake: 0.107 G_L1: 2.909 
End of epoch 675 / 700 	 Time Taken: 0 sec
update learning rate: 0.000017 -> 0.000017
End of epoch 676 / 700 	 Time Taken: 0 sec
update learning rate: 0.000017 -> 0.000016
End of epoch 677 / 700 	 Time Taken: 0 sec
update learning rate: 0.000016 -> 0.000015
End of epoch 678 / 700 	 Time Taken: 0 sec
update learning rate: 0.000015 -> 0.000015
End of epoch 679 / 700 	 Time Taken: 0 sec
update learning rate: 0.000015 -> 0.000014
(epoch: 680, iters: 1365, time: 0.064) G1_GAN: 1.589 G1_L1: 5.517 G1_MSE_gt: 0.491 G1_MSE: 1.765 D1_real: 0.101 D1_fake: 0.096 G_L1: 2.877 
End of epoch 680 / 700 	 Time Taken: 0 sec
update learning rate: 0.000014 -> 0.000013
End of epoch 681 / 700 	 Time Taken: 0 sec
update learning rate: 0.000013 -> 0.000013
End of epoch 682 / 700 	 Time Taken: 0 sec
update learning rate: 0.000013 -> 0.000012
End of epoch 683 / 700 	 Time Taken: 0 sec
update learning rate: 0.000012 -> 0.000011
End of epoch 684 / 700 	 Time Taken: 0 sec
update learning rate: 0.000011 -> 0.000011
(epoch: 685, iters: 1375, time: 0.074) G1_GAN: 1.512 G1_L1: 5.515 G1_MSE_gt: 0.487 G1_MSE: 1.738 D1_real: 0.084 D1_fake: 0.135 G_L1: 2.898 
End of epoch 685 / 700 	 Time Taken: 0 sec
update learning rate: 0.000011 -> 0.000010
End of epoch 686 / 700 	 Time Taken: 0 sec
update learning rate: 0.000010 -> 0.000009
End of epoch 687 / 700 	 Time Taken: 0 sec
update learning rate: 0.000009 -> 0.000009
End of epoch 688 / 700 	 Time Taken: 0 sec
update learning rate: 0.000009 -> 0.000008
End of epoch 689 / 700 	 Time Taken: 0 sec
update learning rate: 0.000008 -> 0.000007
(epoch: 690, iters: 1385, time: 0.069) G1_GAN: 1.597 G1_L1: 5.607 G1_MSE_gt: 0.509 G1_MSE: 1.881 D1_real: 0.088 D1_fake: 0.116 G_L1: 2.903 
End of epoch 690 / 700 	 Time Taken: 0 sec
update learning rate: 0.000007 -> 0.000007
End of epoch 691 / 700 	 Time Taken: 0 sec
update learning rate: 0.000007 -> 0.000006
End of epoch 692 / 700 	 Time Taken: 0 sec
update learning rate: 0.000006 -> 0.000005
End of epoch 693 / 700 	 Time Taken: 0 sec
update learning rate: 0.000005 -> 0.000005
End of epoch 694 / 700 	 Time Taken: 0 sec
update learning rate: 0.000005 -> 0.000004
(epoch: 695, iters: 1395, time: 0.073) G1_GAN: 1.495 G1_L1: 5.525 G1_MSE_gt: 0.491 G1_MSE: 1.578 D1_real: 0.089 D1_fake: 0.101 G_L1: 2.891 
End of epoch 695 / 700 	 Time Taken: 0 sec
update learning rate: 0.000004 -> 0.000003
End of epoch 696 / 700 	 Time Taken: 0 sec
update learning rate: 0.000003 -> 0.000003
End of epoch 697 / 700 	 Time Taken: 0 sec
update learning rate: 0.000003 -> 0.000002
End of epoch 698 / 700 	 Time Taken: 0 sec
update learning rate: 0.000002 -> 0.000001
End of epoch 699 / 700 	 Time Taken: 0 sec
update learning rate: 0.000001 -> 0.000001
(epoch: 700, iters: 1405, time: 0.071) G1_GAN: 1.395 G1_L1: 5.564 G1_MSE_gt: 0.509 G1_MSE: 1.467 D1_real: 0.088 D1_fake: 0.130 G_L1: 2.870 
saving the model at the end of epoch 700, iters 4900
End of epoch 700 / 700 	 Time Taken: 1 sec
update learning rate: 0.000001 -> 0.000000
