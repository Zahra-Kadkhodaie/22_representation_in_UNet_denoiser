import numpy as np
import torchvision
import random
import os
import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# from torch.optim.lr_scheduler import StepLR
sys.path.insert(0, '/mnt/home/zkadkhodaie/projects/shared_code')

from plotting_func import plot_loss


sys.path.insert(0, '../../shared_code')
from model_loader_func import * 
from dataloader_func import *
from quality_metrics_func import *
from linear_approx import *
from algorithm_inv_prob import * 

#########################################################################################################



def main():
    data_name = 'imagenet'
    ######################################################################################################### 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

     
    ### texture 
    if data_name == 'texture':

        phis = torch.load('../results/phis_noise_ave_80x80_mixture-color-no-skip-deep_texture.pt', weights_only= True)


    ### imagenet 
    elif data_name == 'imagenet':

        dir_name = '/mnt/home/zkadkhodaie/ceph/22_representation_in_UNet_denoiser/classifiers/linear/imagenet/'

        
        phis_train = torch.load('../results/phis_noise_ave_trainset_full_mixture-color-no-skip-deep_imagenet64x64.pt', weights_only= True)
        phis_val = torch.load('../results/phis_noise_ave_validation_full_mixture-color-no-skip-deep_imagenet64x64.pt', weights_only= True)


        
    ######################################################################################################### 
    
    sig = 50 
    # --- Configuration ---
    
    batch_size = 2048
    learning_rate = 1e-3
    num_epochs = 10_000

    # train data 
    X = torch.cat(phis_train[sig])[:, 64 + 128 + 256: 64+ 128 + 256 + 512]          # Latent features
    
    latent_dim = X.shape[1]      # Size of your latent vector
    num_classes = len(phis_train[sig])    # Number of classification categories
    
    print(latent_dim, num_classes)

    labels = [torch.ones(len(phis_train[sig][i]) )*i for i in range(len(phis_train[sig])) ]
    
    y = torch.cat(labels)    # Corresponding labels
    y = y.long()
    print(X.shape, y.shape)    
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



    # test data 
    X_val  = torch.cat(phis_val[sig])[:,64 + 128 + 256: 64+ 128 + 256 + 512]          # Latent features
        
    labels_val  = [torch.ones(len(phis_val[sig][i]) )*i for i in range(len(phis_val[sig])) ]
    
    y_val = torch.cat(labels_val)    # Corresponding labels
    y_val = y_val.long()
    print(X_val.shape, y_val.shape) 
    
    dataset_val = TensorDataset(X_val, y_val)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)


    
    
    
    # --- Linear Classifier Model ---
    class LinearClassifier(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.fc = nn.Linear(input_dim, output_dim)
    
        def forward(self, x):
            return self.fc(x)
    
    model = LinearClassifier(latent_dim, num_classes).cuda()


    
    # --- Loss and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # StepLR: decay LR by gamma every step_size epochs
    # scheduler = StepLR(optimizer, step_size=3000, gamma=0.5)
    
    all_acc_train = []
    all_acc_val = []
    epoch_loss_train = []
    epoch_loss_val = []
    
    start_time_total = time.time()        
    
    # --- Training Loop ---
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
    
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
    
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item() * batch_x.size(0)
            correct += (logits.argmax(dim=1) == batch_y).sum().item()
        
        acc = correct / len(dataset)
        all_acc_train.append(acc)
        epoch_loss_train.append(total_loss / len(dataset) ) 


        
        model.eval()
        total_loss = 0
        correct = 0
    
        for batch_x, batch_y in loader_val:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
    
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
    
            total_loss += loss.item() * batch_x.size(0)
            correct += (logits.argmax(dim=1) == batch_y).sum().item()
            
        acc = correct / len(dataset_val)
        all_acc_val.append(acc)
        epoch_loss_val.append(total_loss / len(dataset_val) ) 

        
        plot_loss(epoch_loss_train, epoch_loss_val, dir_name+'/loss_epoch.png')
        plot_loss(all_acc_train, all_acc_val, dir_name+'/accuracy_epoch.png')

        torch.save(model.state_dict(),  dir_name +'/model.pt')
      
        # print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataset):.4f}, Accuracy = {acc:.4f}")
        
        # Step the LR scheduler **after** each epoch
        # scheduler.step()
        # Optional: print learning rate
        # current_lr = scheduler.get_last_lr()[0]
        # print(f"Epoch {epoch+1}, LR: {current_lr:.5f}")
    
    print("--- %s seconds ---" % (round(time.time() - start_time_total)))


if __name__ == "__main__" :
    main()