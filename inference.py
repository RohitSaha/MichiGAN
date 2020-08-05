"""
Copyright (C) University of Science and Technology of China.
Licensed under the MIT License.
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util.util import tensor2im, tensor2label, blend_image
from util import html
from data.base_dataset import single_inference_dataLoad
from PIL import Image
import torch
import math
import numpy as np
import torch.nn as nn
import cv2

opt = TestOptions().parse()

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

criterionRGBL1 = nn.L1Loss()
criterionRGBL2 = nn.MSELoss()

# read data
data = single_inference_dataLoad(opt)
# forward
generated = model(data, mode='inference')
img_path = data['path']
print('process image... %s' % img_path)

# remove background
if opt.remove_background:
    generated = generated * data['label_tag'].float() + data['image_tag'] *(1 - data['label_tag'].float())
fake_image = tensor2im(generated[0])
if opt.add_feat_zeros or opt.add_zeros:
    th = opt.add_th
    H, W = opt.crop_size, opt.crop_size
    fake_image_tmp = fake_image[int(th/2):int(th/2)+H,int(th/2):int(th/2)+W,:]
    fake_image = fake_image_tmp

fake_image_np = fake_image.copy()
img = np.array(Image.open('/home/ubuntu/MichiGAN_FFHQ/val_images/{}.jpg'.format(
    opt.inference_ref_name)).convert("RGB").resize((128, 128)))

lab = np.array(Image.open('/home/ubuntu/MichiGAN_FFHQ/val_labels/{}.png'.format(
    opt.inference_tag_name)).convert("RGB").resize((128, 128)))

orient = np.array(
    Image.open('/home/ubuntu/MichiGAN_FFHQ/val_dense_orients/{}_orient_dense.png'.format(
        opt.inference_orient_name)).convert("RGB").resize((128, 128)))

image = np.concatenate((np.uint8(img), np.uint8(lab), np.uint8(orient), np.uint8(fake_image)),
    axis=1)
image = Image.fromarray(image)

fake_image = Image.fromarray(np.uint8(fake_image))

# val_images - inference_ref_name
# val_labels - inference_tag_name
# val_orient - inference_orient_name

if opt.use_ig:
    #fake_image.save('./inference_samples/inpaint_fake_image.jpg')
    image.save("./inference_samples/inpaint_ref-{}_tag-{}_orient-{}.jpg".format(
        opt.inference_ref_name, opt.inference_tag_name, opt.inference_orient_name))
else:
    fake_image.save('./inference_samples/fake_image.jpg')
