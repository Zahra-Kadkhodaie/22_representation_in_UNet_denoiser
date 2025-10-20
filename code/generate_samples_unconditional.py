import numpy as np
import torchvision
import random

import os
import time
import torch
import sys
sys.path.insert(0, '../../shared_code')
from model_loader_func import * 
from dataloader_func import *
from quality_metrics_func import *
from algorithm_inv_prob import * 



def main():
    data_name = 'multi_class'
    ######################################################################################################### 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    



    ######################################################################################################### 
    ### texture 
    if data_name == 'texture':
        paths = { 
                    'mixture-color-no-skip-deep': 'UNet_flex/texture_EPS_color/0to255_RF_84x84_set_size_237580_color_no_skip_deep_enc_80x80/' , 
                }
        K = 80
    ### imagenet 
    elif data_name == 'imagenet':
        paths = { 
                     'mixture-color-no-skip-deep': 'UNet_flex/imagenet/0to255_RF_84x84_set_size_1232457_color_no_skip_deep_dec_64x64/',
            
            
                }
    
        K = 64
    #### six_class
    #### multi_class
    elif data_name == 'multi_class': 
        paths = { 
             'mixture-color-deep-no-skip' :  'UNet_flex/multi_class_dataset/0to255_RF_84x84_set_size_270000_color_no_skip_deep_dec_80x80/'         
                }

        K = 80
    


    ######################################################################################################### 

    root_path = '/mnt/home/zkadkhodaie/ceph/22_representation_in_UNet_denoiser/denoisers/'

    denoisers = {}
    
    groups = paths.keys()
    for group in groups: 
        path = root_path + paths[group]
    
        print('loading group ' , group )
        denoisers[group] = load_learned_model(path, print_args=True)

        
    
    ######################################################################################################### 

    total_n_samples = 10_000
    n_samples =50


    n_channels= 3
    
    seed = None
    freq =0
    sig_L = .04
    h0= .01
    beta = .05
    sig_0 = 3
    fixed_h =True
    skip = True
    start_time_total = time.time()
    
    for group in groups: 
        if 'skip' in group.split('-'): 
            skip = False
        if seed is not None:
            torch.manual_seed(seed)
            
    
        
        print('--------- group : ', group)        
        temp = []
        for _ in range(int(total_n_samples/n_samples)):    
            if 'color' in group.split('-'): 
                n_channels = 3
                init_im = torch.ones((n_samples,n_channels ,K,K) ,device = device )*.45+ torch.randn(n_samples,n_channels ,K,K, device = device)*sig_0
                
            else: 
                n_channels = 1        
                init_im = dist_mean.mean(dim=0,keepdim=True).to(device)+ torch.randn(n_samples,n_channels ,K,K, device = device) * sig_0
    
            
            sample, _,_, _,t = batch_synthesis(denoisers[group],
                                                  init_im=init_im, 
                                                  sig_L=sig_L, 
                                                  h0=h0, 
                                                  beta=beta, 
                                                  freq=freq,
                                                  device=device, 
                                                  fixed_h = fixed_h,
                                                  max_T=20000, 
                                                  seed= seed, 
                                                  skip = skip
                                                  )
                        
            temp.append(sample.detach())    
        
            all_samples = torch.cat(temp)
        
        

            torch.save(all_samples, '/mnt/home/zkadkhodaie/projects/22_representation_in_UNet_denoiser/results/samples_unconditional_'+str(K)+'x'+str(K)+'_'+group+'_'+data_name+ '.pt' )
    print("--- %s seconds ---" % (time.time() - start_time_total))
    
if __name__ == "__main__" :
    main()
