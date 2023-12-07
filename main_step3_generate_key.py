
import json
import os
import sys
import time
import torch

from data.conversion import GridDataConverter, PointCloudDataConverter, ERA5Converter
from data.dataloaders import mnist, celebahq
from data.dataloaders_era5 import era5
from data.dataloaders3d import shapenet_voxels, shapenet_point_clouds

###############################################################
# # config file for training your stego INR model
# config_path = "configs\config_celebahq64StegaINR.json"
config_path = "configs\config_era5StegaINR.json"
# config_path = "configs\config_shapenet_point_clouds_StegaINR.json"
# config_path = "configs\config_shapenet_voxels_StegaINR.json"

##############################################################
 # dir for your trained secret models which should contain secret inr model file(*.pt) and config.json: (for load secret model)
# secret_model_and_config_dir = 'trained-models/celebahq64_INR'   #
# secret_model_and_config_dir = 'trained-models/celebahq128_INR'
# secret_model_and_config_dir = 'trained-models/celebahq256_INR'
# secret_model_and_config_dir = 'trained-models/celebahq512_INR'

secret_model_and_config_dir = 'trained-models/ear5_INR'
# secret_model_and_config_dir = 'trained-models/shapenet_point_clouds_INR'
# secret_model_and_config_dir = 'trained-models/shapenet_voxels_INR'

#############################################################################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get config file from command line arguments
# if len(sys.argv) != 2:
#     raise(RuntimeError("Wrong arguments, use python main.py <config_path>"))
# config_path = sys.argv[1]

# Open config file
with open(config_path) as f:
    config = json.load(f)

# config_path # already exist for config of stego model
print("stego config path:", config_path)

config_secret_path =secret_model_and_config_dir+"/config.json"
print("secrete model and config dir:",config_secret_path)


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



def generate_key_by_config(config_path, config_secret_path, output_dim):
    with open(config_path) as f:
        config = json.load(f)
    with open(config_secret_path) as f:
        config_secret = json.load(f)
    #get stego model layers
    stego_model_layers = []
    num_stego_frequencies = config["generator"]["encoding"]["num_frequencies"]

    if num_stego_frequencies == 0:   # for cloud points and voxel
        num_stego_firslayer = input_dim
    else:
        num_stego_firslayer = num_stego_frequencies * 2

    stego_model_layers.append(num_stego_firslayer)

    stego_model_hidden_layers = config["generator"]["layer_sizes"]
    stego_model_layers = stego_model_layers + stego_model_hidden_layers

    stego_model_output_layer = output_dim
    stego_model_layers.append(stego_model_output_layer)

    # get secret message model layers
    secret_model_layers = []
    num_secret_frequencies = config_secret["generator"]["encoding"]["num_frequencies"]

    if num_secret_frequencies == 0:   # for cloud points and voxel
        num_secret_firslayer = input_dim
    else:
        num_secret_firslayer = num_secret_frequencies * 2

    secret_model_layers.append(num_secret_firslayer)

    secret_model_hidden_layers = config_secret["generator"]["layer_sizes"]
    secret_model_layers = secret_model_layers+secret_model_hidden_layers

    secret_model_output_layer = output_dim    #same as the stego
    secret_model_layers.append(output_dim)

    # Generate Key with shape of node
    # need to set number of digital for rand generation
    num_rands_number = 10000  # should be lage enough
    key_vector = []
    if len(secret_model_layers)==len(stego_model_layers):
        for T_secret, T_stego in zip(secret_model_layers, stego_model_layers):
            bias_stego = torch.zeros(T_stego)
            bias_secret = torch.ones(T_secret)
            # size = (1,2000) is the sample number that must  enough  large  for paddiing one
            y = torch.randint(0,  bias_stego.shape[0], size=(1,num_rands_number))
            num=0
            x = y[0]
            for i in x:
                # print(i)
                if ( bias_stego[i]==0) and (num <  bias_secret.shape[0]):
                    bias_stego[i]=1
                    num = num+1
            # tensor to list for save json
            bias_stego_list = bias_stego.tolist()
            key_vector.append(bias_stego_list)
        # save key vector to a json file
        json_key_data = json.dumps(key_vector)
        with open("key.json", "w") as file:  # make sure key.json is not exist
            file.write(json_key_data)
        print("key generation complete!")
        return key_vector

    elif len(secret_model_layers)<len(stego_model_layers):
        count_secret_layer = 0
        for T_stego in stego_model_layers:
              # layer count for secret model
            bias_stego = torch.zeros(T_stego)
            if count_secret_layer < len(secret_model_layers):
                bias_secret= torch.ones(secret_model_layers[count_secret_layer])

                y = torch.randint(0, bias_stego.shape[0], size=(1, num_rands_number))
                num = 0
                x = y[0]
                for i in x:
                    # print(i)
                    if (bias_stego[i] == 0) and (num < bias_secret.shape[0]):
                        bias_stego[i] = 1
                        num = num + 1
                count_secret_layer = count_secret_layer+1

            bias_stego_list = bias_stego.tolist()

            key_vector.append(bias_stego_list)

        # save key vector to a json file
        json_key_data = json.dumps(key_vector)
        with open("key.json", "w") as file:  # make sure key.json is not exist
            file.write(json_key_data)
        print("key generation complete!")
        return key_vector

    if len(secret_model_layers) > len(secret_model_layers):
        print("Error! number of layers for secret model  should larger than  umber of layers for secret model ")
        return key_vector


generate_key_by_config(config_path, config_secret_path, output_dim)