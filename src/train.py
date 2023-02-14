import argparse
import json
import os

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import config
import wandb
from custom_data import CustomData
from model import ChatBot
from utils import init_wandb, plot


def data_loader(params):
    data = json.load(open(params["ENCODED_DATA"],"r"))
    print(f"Total number of samples : {len(data)}\n")
    test_size = round(len(data)*params["TEST_SIZE"])
    train_data = data[test_size:]
    validation_data = data[:test_size]

    train_dataset = CustomData(train_data)
    validation_dataset = CustomData(validation_data)

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=config.BATCH_SIZE,shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_dataset,batch_size=config.BATCH_SIZE,shuffle=False)
    print(f"Traning data size : {len(train_dataset)}")
    print(f"Validation data size : {len(validation_dataset)}\n")
    return train_loader,val_loader

def run_batch(params,loader,device, model,epoch,type = "train"):
    optimizer = Adam(model.parameters(), lr=params["LEARNING_RATE"])
    scheduler = StepLR(optimizer, step_size= params['STEP_SIZE'], gamma=params["GAMMA"])
    model.train()
    total_loss = 0
    loader_tqdm = tqdm(enumerate(loader),total = len(loader),desc = f"Epoch {epoch}: ")
    for _,data in loader_tqdm:
        input_sequence = data['input_sequence'].to(device)
        output_sequence = data['output_sequence'].to(device)
        # length = data['input_length']
        optimizer.zero_grad(set_to_none=True)
        _,loss = model(input_sequence,output_sequence)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loader_tqdm.set_description_str(type.title())
        loader_tqdm.set_postfix_str(f"{type.title()} LOSS: {loss.item():.2f}")
    learning_rate = optimizer.param_groups[0]['lr']
    scheduler.step()
    return total_loss,learning_rate

@torch.no_grad()
def run_eval_batch(params,loader,device, model,epoch,type = "validation"):
    model.eval()
    total_loss = 0
    loader_tqdm = tqdm(enumerate(loader),total = len(loader),desc = f"Epoch {epoch}: ")
    for _,data in loader_tqdm:
        input_sequence = data['input_sequence'].to(device)
        output_sequence = data['output_sequence'].to(device)
        _,loss = model(input_sequence,output_sequence)
        total_loss += loss.item()
        loader_tqdm.set_description_str(type.title())
        loader_tqdm.set_postfix_str(f"{type.title()} LOSS: {loss.item():.2f}")
    return total_loss

def run_epoch(arguments,params,model):
    if arguments.device:
        device = arguments.device
    else:
        device = params["DEVICE"]
    print(f"Using {device} Device")
    model.to(device)
    model.train()
    train_loader,val_loader = data_loader(params)
    epoch_tqdm_ob = tqdm(range(params['EPOCHS']),total = params['EPOCHS'],desc = f"Traning Model on {params['EPOCHS']} Epochs : ")
    train_losses = list()
    val_losses = list()
    epochs = list(range(1,params['EPOCHS']+1))
    val_loss = np.inf
    for epoch in epoch_tqdm_ob:
        train_total_loss,learning_rate = run_batch(params,train_loader,device,model,epoch,type="train")
        val_total_loss = run_eval_batch(params,val_loader,device,model,epoch,type="val")
        train_losses.append(train_total_loss)
        val_losses.append(val_total_loss)
        epoch_tqdm_ob.set_description(f"Epoch: {epoch+1:.4f}, Loss: {train_total_loss / len(train_loader):.4f}")
        epoch_tqdm_ob.set_description(f"Epoch: {epoch+1:.4f}, Loss: {val_total_loss / len(val_loader):.4f}")
        directory = "checkpoints/" + params["MODEL_NAME"] 
        if not os.path.exists(directory):
            os.makedirs(directory)
        if val_total_loss < val_loss:
            val_loss = val_total_loss
            early_stopping = 0
            torch.save(
                {
                    "iteration" : epoch,
                    "encoder" : model.encoder.state_dict(),
                    "decoder" : model.decoder.state_dict(),
                    "model_state_dict" : model.state_dict(),
                    "train_losses" : train_total_loss,
                    "validation_losses": val_losses,
                    "params" : params,
                    "argument_params" : arguments,
                    "device" : device,
                    "learning_rate":learning_rate
                },
                os.path.join(directory,f'{params["MODEL_NAME"]}_batch_size_{params["BATCH_SIZE"]}_hidden_size{params["HIDDEN_SIZE"]}.pt')
            )
        else:
            early_stopping += 1
        if early_stopping == params['PATIENCE']:
            print(f"EARLY STOPPING AT EPOCH {epoch+1}")
            break
        wandb.log({"epoch": epoch, "training_loss": train_total_loss, "val_loss": val_total_loss,"learning_rate":learning_rate})
    plot(epochs,[train_losses,val_losses])
if __name__=="__main__":
    params = {k:v for k,v in config.__dict__.items() if "__" not in k}
    vocab = json.load(open("dataset/vocab.json","r"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",choices=["cpu","cuda"])
    parser.add_argument("--wandb",choices=["online","disabled"],default="disabled")
    arguments = parser.parse_args()
    vocab_size = len(vocab)
    model = ChatBot(params,vocab_size)
    init_wandb(params,arguments)
    run_epoch(arguments,params,model)


    
