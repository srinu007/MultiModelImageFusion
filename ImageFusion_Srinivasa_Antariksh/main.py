
from __future__ import print_function

import time

from train_recons import train_recons
from generate import generate
from utils import list_images
import os
IS_TRAINING = False
is_RGB = True

BATCH_SIZE = 2
EPOCHES = 4

SSIM_WEIGHTS = [1, 10, 100, 1000]
MODEL_SAVE_PATHS = [
    './models/densefuse_gray/densefuse_model_bs2_epoch4_all_weight_1e0.ckpt',
    './models/densefuse_gray/densefuse_model_bs2_epoch4_all_weight_1e1.ckpt',
    './models/densefuse_gray/densefuse_model_bs2_epoch4_all_weight_1e2.ckpt',
    './models/densefuse_gray/densefuse_model_bs2_epoch4_all_weight_1e3.ckpt',
]

model_pre_path = None

def main():

	if IS_TRAINING:

		original_imgs_path = list_images('D:/Database/Image_fusion_MSCOCO/original/')
		validatioin_imgs_path = list_images('./validation/')

		for ssim_weight, model_save_path in zip(SSIM_WEIGHTS, MODEL_SAVE_PATHS):
			print('\nBegin to train the network ...\n')
			train_recons(original_imgs_path, validatioin_imgs_path, model_save_path, model_pre_path, ssim_weight, EPOCHES, BATCH_SIZE, debug=True)

			print('\nSuccessfully! Done training...\n')
	else:
			ssim_weight = SSIM_WEIGHTS[2]
			model_path = MODEL_SAVE_PATHS[2]
			print('\nBegin to generate pictures ...\n')
			
			path = 'images/MF_images/color/'
			for i in range(1):
				index = i + 1
				
				infrared = path + 'b1.jpg'
				visible = path + 'b2.jpg'

				# choose fusion layer
				fusion_type = 'addition'
				output_save_path = 'outputs'
				generate(infrared, visible, model_path, model_pre_path,
						 ssim_weight, index, False, is_RGB, type = fusion_type, output_path = output_save_path)


if __name__ == '__main__':
    main()

