import sys
import warnings
import time
from tqdm import tqdm, trange
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
import torch.optim as optim
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.dataloader import get_dataloaders
from utils.checkpoint import save
from utils.running import train, evaluate
from utils.setup_network import setup_network
from utils.helper import epoch_time, store_folder

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

if torch.cuda.is_available():
    torch.cuda.manual_seed(123)
    torch.cuda.synchronize() 
else:
    torch.manual_seed(123)
np.random.seed(123)

# CUDA_VISIBLE_DEVICES=1 python train_lr_opt.py --bs 64 --target_size 48 --num_epochs 300 --lr 0.01 
# --optimizer AdamW --network finalv2 --data_name fer_plus --training_mode new > fer_plus.txt

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True, default="fer2013")
    parser.add_argument("--bs", type=int, required=True, default=64,
                        help="Batch size of model")
    parser.add_argument("--target_size", type=int, required=True, default=48,
                        help="Image target size to resize")
    parser.add_argument("--num_epochs", type=int, required=True, default=300,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, required=True, default=0.1,
                        help="Learning rate")
    parser.add_argument("--training_mode", type=str, required=True, default="search",
                        help="Training model is defined to search params or not")
    parser.add_argument("--optimizer", type=str, required=True, default="SGD",
                        help="Optimizer to update weights")
    parser.add_argument("--network", type=str, required=True, default="cbam_resmob",
                        help="Name of network")
    parser.add_argument("--model_path", type=str, default="./models",
                        help="Path to models folder that contains many models")
    parser.add_argument("--dataset_path", type=str, default="/home/aime/hoangdh/emotions/dataset")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--es_min_delta", type=float, default=0.0)
    parser.add_argument("--es_patience", type=int, default=5)
    parser.add_argument("--root_path", type=str,default="genki4k_data")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    args = vars(parser.parse_args())
    
    return args

def run(net, 
        logger, 
        dataset_path, 
        data_name, 
        batch_size, 
        target_size, 
        optimizer, 
        learning_rate, 
        num_epochs, 
        model_save_dir,
        parameters_config=[]
        ):
    
    print('batch_size: ', batch_size)
    print('learning_rate: ', learning_rate)
    print('optimizer: ', optimizer)
    data_path = os.path.join(dataset_path, data_name)
    phases = os.listdir(data_path)
    data_loaders = get_dataloaders(data_path,
                                   phases=phases,
                                   target_size=target_size, 
                                   batch_size=batch_size)
    if 'train' not in data_loaders:
        raise ValueError('train data not found')
    if len(data_loaders) == 1:
        raise ValueError('please add val or test data')
    train_dataloader = data_loaders['train']
    if 'val' in phases:
        val_dataloader = data_loaders['val']
    if 'test' in phases:
        test_dataloader = data_loaders['test']
    net = net.to(device)
    # Scaler
    scaler = GradScaler()

    # Optimizer
    if len(parameters_config) == 0:
        optimizer = getattr(optim, optimizer)(net.parameters(), lr= learning_rate)
        if optimizer == 'SGD':
            optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=0.001)
    else:
        optimizer = getattr(optim, optimizer)(parameters_config)
        if optimizer == 'SGD':
            optimizer = torch.optim.SGD(parameters_config, momentum=0.9, nesterov=True, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=5, verbose=True)
    
    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)   
    
    best_acc = 0.0
    
    for epoch in trange(num_epochs, desc="Epochs"):
        start_time = time.monotonic()
        loss_train, acc_train, f1_train = train(net, train_dataloader, criterion, optimizer, scaler)
        logger.loss_train.append(loss_train)
        logger.acc_train.append(acc_train)
        
        if 'val' in phases:
            loss_val, acc_val, f1_val = evaluate(net, val_dataloader, criterion)
            logger.loss_val.append(loss_val)
            logger.acc_val.append(acc_val)

            # if learning_rate >= 0.01:
            scheduler.step(acc_val)
            
            end_time = time.monotonic()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            # save best checkpoint
            if acc_val > best_acc:
                best_acc = acc_val
                save(net, logger, model_save_dir, "best")
                logger.save_plt(model_save_dir)
        
        # save last checkpoint (frequency)
        save(net, logger, model_save_dir, "last")
        logger.save_plt(model_save_dir)
            
        print(f'epoch: {epoch+1:02} | epoch time: {epoch_mins}m {epoch_secs}s')
        print(f'\ttrain loss: {loss_train:.3f} | train acc: {acc_train*100:.2f}% | train F1: {f1_train*100:.2f}%')
        if 'val' in phases:
            print(f'\t val loss: {loss_val:.3f} |  val acc: {acc_val*100:.2f}% | val F1: {f1_val*100:.2f}%')
    if 'test' in phases:       
        _ = evaluate(net, test_dataloader, criterion, "Testing")

    return best_acc
  
if __name__ == "__main__":
    args = get_args()
    available_nets = set(filename.split(".")[0] for filename in os.listdir(args["model_path"]))
    available_datasets = set(filename for filename in os.listdir(args["dataset_path"]))
    in_channels = 1
    if "FERG" in args["data_name"]:
        in_channels = 3
    model_save_dir = store_folder(args["data_name"], args["network"], args["bs"], args["optimizer"], args["lr"], available_nets)
    logger, net = setup_network(args["network"], in_channels)
    best_acc = run(net, logger, args["dataset_path"], args["data_name"], args["bs"], args["target_size"], args["optimizer"], 
                    args["lr"], args["num_epochs"], model_save_dir)