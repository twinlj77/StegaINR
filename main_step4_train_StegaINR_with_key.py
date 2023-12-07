import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"  # sovle the multiple copies of libiomp5md.dll problem

import sys
import time
import torch
# from training.training import Trainer

# modify trainning.py for INR
# from training.training_INR import TrainerINR
from training.training_INR_Stego import TrainerINRStego

from data.conversion import GridDataConverter, PointCloudDataConverter, ERA5Converter
from data.dataloaders import mnist, celebahq
from data.dataloaders_era5 import era5
from data.dataloaders3d import shapenet_voxels, shapenet_point_clouds
from models.discriminator import PointConvDiscriminator
from models.function_distribution import HyperNetwork, FunctionDistribution
from models.function_representation import FunctionRepresentation, FourierFeatures

# Add for INR：load inr model for function representation
from models.function_representation import load_function_representation


########################################################################
# dir for loading secret function model to train stegaINR model (should contain secret inr model file(*.pt) and config.json)
# secret_model_and_config_dir = 'trained-models/celebahq64_INR'
# secret_model_and_config_dir = 'trained-models/celebahq128_INR'
# secret_model_and_config_dir = 'trained-models/celebahq256_INR'
# secret_model_and_config_dir = 'trained-models/celebahq512_INR'

secret_model_and_config_dir = 'trained-models/ear5_INR'
# secret_model_and_config_dir = 'trained-models/shapenet_point_clouds_INR'
# secret_model_and_config_dir = 'trained-models/shapenet_voxels_INR'

########################################################################
# config file for train StegaINR model
# config_path = "configs\config_celebahq64StegaINR.json"
config_path = "configs\config_era5StegaINR.json"
# config_path = "configs\config_shapenet_point_clouds_StegaINR.json"
# config_path = "configs\config_shapenet_voxels_StegaINR.json"


#################################################################################################################################################################

def load_secret_function_model(exp_dir):
    # exp_dir = 'trained-models/celebahq64_INR'  # for your trained models folder

    # config.json should be consistent with config.json in experiment folder
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

    # Load function distribution weights
    # func_dist = load_function_distribution(device, exp_dir + '/model_250.pt')

    # Add for INR find model format
    files = os.listdir(exp_dir)
    model_files = []
    for file in files:
        #  json format
        if file.endswith('.pt'):
            # json format path
            model_file_path = os.path.join(exp_dir, file)
            model_files.append(model_file_path)

    model_file = model_files[-1]
    print("Load model:", model_file_path)

    # func_dist = load_function_representation(device, model_file)  # one model at least
    func_inr = load_function_representation(device, model_file)
    return func_inr

def print_model(model):
    blank = ' '
    print('-----------------------------------------------')
    print('|   weight name   |        weight shape       |')
    print('-----------------------------------------------')

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 15: key = key + (15 - len(key)) * blank
        w_variable_blank = ''
        if len(w_variable.shape) == 1:
            if w_variable.shape[0] >= 100:
                w_variable_blank = 8 * blank
            else:
                w_variable_blank = 9 * blank
        elif len(w_variable.shape) == 2:
            if w_variable.shape[0] >= 100:
                w_variable_blank = 2 * blank
            else:
                w_variable_blank = 3 * blank

        print('| {} | {}{} |'.format(key, w_variable.shape, w_variable_blank))
        key = 0
    print('-----------------------------------------------')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# os.environ['CUDA_VISIBLE_DEVICES']='1' #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get config file from command line arguments
# if len(sys.argv) != 2:
#     raise(RuntimeError("Wrong arguments, use python main.py <config_path>"))
# config_path = sys.argv[1]

# Open config file
with open(config_path) as f:
    config = json.load(f)

if config["path_to_data"] == "":
    raise(RuntimeError("Path to data not specified. Modify path_to_data attribute in config to point to data."))

# Create a folder to store experiment results
timestamp = time.strftime("%Y-%m-%d_%H-%M")
directory = "{}_{}".format(timestamp, config["id"])
if not os.path.exists(directory):
    os.makedirs(directory)

# Save config file in experiment directory
with open(directory + '/config.json', 'w') as f:
    json.dump(config, f)

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
    frequency_matrix = torch.normal(mean=torch.zeros(num_frequencies, input_dim),
                                    std=std_dev).to(device)
    encoding = FourierFeatures(frequency_matrix)
else:
    encoding = torch.nn.Identity()

# Setup generator models
final_non_linearity = torch.nn.Tanh()
non_linearity = torch.nn.LeakyReLU(0.1)

#Add for INR:
# Setup stego function
function_representation_stego = FunctionRepresentation(input_dim, output_dim,
                                                config["generator"]["layer_sizes"],
                                                encoding, non_linearity,
                                                final_non_linearity).to(device)




function_representation_secret = load_secret_function_model(secret_model_and_config_dir)


print_model(function_representation_stego)
print_model(function_representation_secret)

function_representation_stego_bais_shape = function_representation_stego.get_weight_shapes()[1]
function_representation_secret_bais_shape = function_representation_secret.get_weight_shapes()[1]


# Add 4 INR : just for image type with same output dim
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

def parameters_expand_generate_by_key_and_secret_model(key_json_file,function_representation_secret,function_representation_stego):
    with open(key_json_file) as f:
        key = json.load(f)

    weights_mask, biases_mask  = parameters_mask_generate_by_key(key_json_file)

    with torch.no_grad():
        #secret model parameter
        secret_weights, secret_bias = function_representation_secret.get_weights_and_biases()

        #stego_model_parameter
        stego_weights, stego_bias = function_representation_stego.get_weights_and_biases()

       # expand bias
        biases_expand = biases_mask.copy()
        for T_secret, T_mask, T_stego in zip(secret_bias, biases_expand, stego_bias):
            T_stego_mask_location = [i for i, e in enumerate(T_mask) if e != 0]
            for iterm_value, location_index in zip(T_secret, T_stego_mask_location):
                # print(T_stego[location_index])
                T_stego[location_index].requires_grad = False
                T_stego[location_index] = iterm_value
                # print(T_stego[location_index])


        # stego weights_mask
        weights_expand = weights_mask.copy()
        for T_secret, T_mask, T_stego  in zip(secret_weights, weights_expand, stego_weights):

            locate_item = []
            for i, item in zip(range(len(T_mask)), T_mask):
                for j, e in enumerate(item):
                    if e != 0:
                        index = [i, j]
                        locate_item.append(index)

            mask_sq_index =0
            for i in range(len(T_secret)):
                for j in range(len(T_secret[0])):
                    locate_index = locate_item[mask_sq_index]
                    # print(T_stego[locate_index[0],locate_index[1]])
                    T_stego[locate_index[0],locate_index[1]]=T_secret[i,j]
                    T_stego[locate_index[0], locate_index[1]].requires_grad = False
                    # print(T_stego[locate_index[0],locate_index[1]])
                    mask_sq_index=mask_sq_index+1

    return stego_weights, stego_bias

stego_weights, stego_bias = parameters_expand_generate_by_key_and_secret_model("key.json", function_representation_secret,function_representation_stego)

secret_weights, secret_bias = function_representation_secret.get_weights_and_biases()

function_representation_stego.set_weights_and_biases(stego_weights, stego_bias)

stego_weights_modifed, secret_bias_modified = function_representation_stego.get_weights_and_biases()

# for item_o, item_t , item_m in zip(stego_bias, secret_bias, secret_bias_modified):
#     print("stego_bias:", item_o)
#     print("secret_bias:", item_t)
#     print("stego_bias_modified:", item_m)


############################################
# Add for INR:
print("\nFunction Representation")
print(function_representation_stego)
print("Number of parameters: {}".format(count_parameters(function_representation_stego)))

###############################################

# modify Setup: Trainer----->TrainerINR
trainer = TrainerINRStego(device, function_representation_stego, data_converter,
                  lr=config["training"]["lr"], lr_disc=config["training"]["lr_disc"],
                  r1_weight=config["training"]["r1_weight"],
                  max_num_points=config["training"]["max_num_points"],
                  print_freq=config["training"]["print_freq"], save_dir=directory,
                  model_save_freq=config["training"]["model_save_freq"],
                  is_voxel=is_voxel, is_point_cloud=is_point_cloud,
                  is_era5=is_era5)
trainer.train(dataloader, config["training"]["epochs"])

