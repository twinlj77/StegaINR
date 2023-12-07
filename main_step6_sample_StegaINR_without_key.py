# -*- coding:utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"  # sovle the multiple copies of libiomp5md.dll problem

import imageio
import json
import torch
from viz.plots import plot_point_cloud_batch, plot_voxels_batch
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
# %matplotlib inline

#Add for INR

from data.conversion import GridDataConverter, PointCloudDataConverter, ERA5Converter
# Add for INR: this import is necessary for load_function_representation
from models.function_representation import load_function_representation

#################################################
#your trained stega models folder, which should cotain model file(*.pt) and config.json
# exp_dir = 'trained-models/celebahq64_StegaINR'
# exp_dir = 'trained-models/shapenet_point_clouds_StegaINR'
# exp_dir = 'trained-models/shapenet_voxels_StegaINR'
exp_dir = 'trained-models/ear5_StegaINR'

#################################################
datatypes=["image","voxel","point_cloud","era5"]
save_dir="."
datatype = datatypes[3]
###################################################################################################

device = torch.device('cpu')
#config.json should be consistent with config.json in experiment folder
with open(exp_dir + '/config.json') as f:
    config = json.load(f)

# Create appropriate data converter based on config
if config["dataset"] == 'mnist':
    data_shape = (1, config["resolution"], config["resolution"])
    data_converter = GridDataConverter(device, data_shape,
                                       normalize_features=True)
elif config["dataset"] == 'celebahq':
    data_shape = (3, config["resolution"], config["resolution"])
    data_converter = GridDataConverter(device, data_shape,
                                       normalize_features=True)
elif config["dataset"] == 'shapenet_voxels':
    data_shape = (1, config["resolution"], config["resolution"], config["resolution"])
    data_converter = GridDataConverter(device, data_shape,
                                       normalize_features=True)
elif config["dataset"] == 'shapenet_point_clouds':
    data_shape = (1, config["resolution"], config["resolution"], config["resolution"])
    data_converter = PointCloudDataConverter(device, data_shape,
                                       normalize_features=True)
elif config["dataset"] == 'era5':
    data_shape = (46, 90)
    data_converter = ERA5Converter(device, data_shape,
                                       normalize_features=True)

# Add for INR find model format
files = os.listdir(exp_dir)
model_files = []
for file in files:
    #  json format
    if file.endswith('.pt'):
        # json format path
        model_file_path = os.path.join(exp_dir, file)
        model_files.append( model_file_path)

model_file = model_files[-1]
print("Load model:", model_file_path)

func_inr = load_function_representation(device, model_file)
# Sample from model

func_inr.save_data_samples_from_representation("sample_from_stego_inr_for_cover.png", data_converter,datatype, save_dir)
print("Sample from stego inr model without key complete !  " )
print("filename: sample_from_stego_inr_for_cover.png")