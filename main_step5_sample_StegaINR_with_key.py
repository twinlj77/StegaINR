# -*- coding:utf-8 -*-

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"  # sovle the multiple copies of libiomp5md.dll problem

import imageio
import json
import torch
from viz.plots import plot_point_cloud_batch, plot_voxels_batch
from torchvision.utils import make_grid

import matplotlib.pyplot as plt


from data.conversion import GridDataConverter, PointCloudDataConverter, ERA5Converter
from data.dataloaders import mnist, celebahq
from data.dataloaders_era5 import era5
from data.dataloaders3d import shapenet_voxels, shapenet_point_clouds
from models.discriminator import PointConvDiscriminator
from models.function_distribution import HyperNetwork, FunctionDistribution
from models.function_representation import FunctionRepresentation, FourierFeatures

# %matplotlib inline

#Add for INR


device = torch.device('cpu')


from data.conversion import GridDataConverter, PointCloudDataConverter
# Note that this import is necessary for load_function_distribution
# properly instantiate the FourierFeatures
from models.function_representation import FourierFeatures
from models.function_distribution import load_function_distribution

# Add for INR: this import is necessary for load_function_representation

from models.function_representation import load_function_representation

#just modified these path for your sampling
# ###################################################
# secret models 's config_file for initial secret inr function (just using config file)
# exp_dir_secret = 'trained-models/celebahq64_INR'
# exp_dir_secret = 'trained-models/celebahq256_INR'
# exp_dir_secret = 'trained-models/celebahq512_INR'

# exp_dir_secret = 'trained-models/shapenet_point_clouds_INR'
# exp_dir_secret = 'trained-models/shapenet_voxels_INR'
exp_dir_secret = 'trained-models/ear5_INR'

# ###################################################
save_dir="."  # sample data save folder, default current
# choose datatypes for save_data_samples_from_representation
datatypes = ["image", "voxel", "point_cloud", "era5"]
datatype = datatypes[3]     # celebaq:0, voxel: 1, point_cloud:2, era5: 3
# ###################################################
#  trained stego models and config  folder for load trained stego inr function
# exp_dir_stego = 'trained-models/celebahq64_StegaINR'

# exp_dir_stego = 'trained-models/shapenet_point_clouds_StegaINR'
# exp_dir_stego = 'trained-models/shapenet_voxels_StegaINR'
exp_dir_stego = 'trained-models/ear5_StegaINR'

############################################################################################
#
# Step 1 : load a stego_function_representation by stego_inr_model_file and config_file
#
############################################################################################
#config.json should be consistent with config.json in experiment folder
with open(exp_dir_stego + '/config.json') as f:
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

# Add for INR find model format, load inr stego model
files = os.listdir(exp_dir_stego)
model_files = []
for file in files:
    #  json format
    if file.endswith('.pt'):
        # json format path
        model_file_path = os.path.join(exp_dir_stego, file)
        model_files.append(model_file_path)

model_file = model_files[-1]

print("Load stego model:", model_file_path)
func_inr_stego = load_function_representation(device, model_file) # one model at least
# Sample from model
# datatype=["image", "voxel", "point_cloud", "era5"]
# save_dir="."
# func_inr_stego.save_data_samples_from_representation("sampled-inr-stego-image.png", data_converter,datatype[0], save_dir)

############################################################################################
#
# Step 2 : initial an extracted secret_function_representation by config_file
#
############################################################################################
# exp_dir_secret = 'trained-models/celebahq64_INR'
with open(exp_dir_secret + '/config.json') as f:
    config = json.load(f)
# Setup dataloader
is_voxel = False
is_point_cloud = False
is_era5 = False

if config["dataset"] == 'mnist':
    dataloader = mnist(path_to_data=config["path_to_data"],
                       batch_size=config["training"]["batch_size"],
                       size=config["resolution"],
                       train=True)
    input_dim = 2
    output_dim = 1
    data_shape = (1, config["resolution"], config["resolution"])
elif config["dataset"] == 'celebahq':
    dataloader = celebahq(path_to_data=config["path_to_data"],
                          batch_size=config["training"]["batch_size"],
                          size=config["resolution"])
    input_dim = 2
    output_dim = 3
    data_shape = (3, config["resolution"], config["resolution"])
elif config["dataset"] == 'shapenet_voxels':
    dataloader = shapenet_voxels(path_to_data=config["path_to_data"],
                                 batch_size=config["training"]["batch_size"],
                                 size=config["resolution"])
    input_dim = 3
    output_dim = 1
    data_shape = (1, config["resolution"], config["resolution"], config["resolution"])
    is_voxel = True
elif config["dataset"] == 'shapenet_point_clouds':
    dataloader = shapenet_point_clouds(path_to_data=config["path_to_data"],
                                       batch_size=config["training"]["batch_size"])
    input_dim = 3
    output_dim = 1
    data_shape = (1, config["resolution"], config["resolution"], config["resolution"])
    is_point_cloud = True
elif config["dataset"] == 'era5':
    dataloader = era5(path_to_data=config["path_to_data"],
                      batch_size=config["training"]["batch_size"])
    input_dim = 3
    output_dim = 1
    data_shape = (46, 90)
    is_era5 = True

# Setup data converter : convert original data to coordinates and features.
if is_point_cloud:
    data_converter = PointCloudDataConverter(device, data_shape, normalize_features=True)
elif is_era5:
    data_converter = ERA5Converter(device, data_shape, normalize_features=True)
else:
    data_converter = GridDataConverter(device, data_shape, normalize_features=True)

# Setup encoding for function distribution
num_frequencies = config["generator"]["encoding"]["num_frequencies"]
std_dev = config["generator"]["encoding"]["std_dev"]
if num_frequencies:
    # Attention: This is important for using frequency_matrix_for_inr_secret.pt.
    # It should be generated using secret seed by shared by two part
    frequency_matrix = torch.load('frequency_matrix_for_inr_secret.pt')
    encoding = FourierFeatures(frequency_matrix)
else:
    encoding = torch.nn.Identity()

# Setup generator models
final_non_linearity = torch.nn.Tanh()
non_linearity = torch.nn.LeakyReLU(0.1)

func_secret_inr_extracted = FunctionRepresentation(input_dim, output_dim,
                                                config["generator"]["layer_sizes"],
                                                encoding, non_linearity,
                                                final_non_linearity).to(device)

############################################################################################
#
# Step 3 : extract secret_function_ by function_representation_stego and key
#
############################################################################################


def parameters_mask_generate_by_key(key_json_file):
    with open(key_json_file) as f:
        key = json.load(f)

    # biases: just remove input layer
    biases_mask = key.copy()
    biases_mask.pop(0)   #remove input layer

    # convert list to tensor list
    biases_mask_tensor_list = []
    for item in biases_mask:
        biases_mask_tensor_list.append(torch.Tensor(item))

    weights_mask_tensor_list=[]
    for layer, pre_layer in zip(biases_mask, key):
        m = len(layer)
        n = len(pre_layer)
        weight_mask = torch.zeros([m, n])
        for i in range(m):
            for j in range(n):
                if layer[i]==1 and pre_layer[j]== 1:
                    weight_mask[i, j] = 1
        weights_mask_tensor_list.append(weight_mask)

    return weights_mask_tensor_list, biases_mask_tensor_list

def parameters_extract_generate_by_key_and_stego_model(key_json_file,function_representation_secret,function_representation_stego):
    with open(key_json_file) as f:
        key = json.load(f)

    weights_mask, biases_mask  = parameters_mask_generate_by_key(key_json_file)

    with torch.no_grad():
        #secret model parameter
        secret_weights, secret_biases = function_representation_secret.get_weights_and_biases()

        #stego_model_parameter
        stego_weights, stego_bias = function_representation_stego.get_weights_and_biases()

       # extract biases
        biases_expand = biases_mask.copy()
        set_secret_biases =[]
        for T_secret, T_mask, T_stego in zip(secret_biases, biases_expand, stego_bias):
            T_stego_mask_location = [i for i, e in enumerate(T_mask) if e != 0]
            T_secret_biases = T_stego[T_stego_mask_location]
            set_secret_biases.append(T_secret_biases)

        # extract weights
        weights_expand = weights_mask.copy()
        set_secret_weights = []
        for T_secret, T_mask, T_stego  in zip(secret_weights, weights_expand, stego_weights):
            T_mask_bool = T_mask.ge(0.5)
            set_secret_weight  = torch.masked_select(T_stego, T_mask_bool)
            set_secret = set_secret_weight.reshape(-1, len(T_secret[0]))
            set_secret_weights.append(set_secret)
        # set_secret_weights.append(T_secret_weights)
        function_representation_secret.set_weights_and_biases(set_secret_weights, set_secret_biases)


    return set_secret_weights, set_secret_biases

key_json_file = "key.json"
# extract parameters from stego model
secret_extract_weights, secret_extract_biases = parameters_extract_generate_by_key_and_stego_model(key_json_file, func_secret_inr_extracted, func_inr_stego)

# set parameters for secret  model
func_secret_inr_extracted.set_weights_and_biases(secret_extract_weights, secret_extract_biases)

secret_weights, secret_biases = func_secret_inr_extracted.get_weights_and_biases()

# used for check extracted parameters are correct
# model_file_path = 'trained-models/celebahq64_INR/inr_model_950.pt'
# print("Load model:",model_file_path)
# func_inr_original = load_function_representation(device, model_file_path)
# secret_original_weights, secret_original_biases = func_inr_original.get_weights_and_biases()
#
# for item_e, item_o , item in zip(secret_extract_biases, secret_original_biases, secret_biases):
#     print("extracted biases:", item_e)
#     print("originial biases:", item_o)
#     print("modified biases:", item)

func_secret_inr_extracted.save_data_samples_from_representation("sample_from_stego_inr_for_secret_message.png", data_converter,datatype, save_dir)
print("Sample from stego inr model with key complete !  " )
print("filename: sample_from_stego_inr_for_secret_message.png")
