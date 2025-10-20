import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import torchvision
import matplotlib.patches as patches
import random

import os
import time
import torch
import sys
sys.path.insert(0, '../../shared_code')
from model_loader_func import * 
from dataloader_func import load_nested_dataset
from dataloader_func import *
from quality_metrics_func import *
from linear_approx import *
from algorithm_inv_prob import * 



def main():
    data_name = 'imagenet'
    ######################################################################################################### 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    print(torch.cuda.get_device_name(0))


    ######################################################################################################### 
    ### texture 


    ### imagenet 
    if data_name == 'imagenet':
        paths = { 
                     # 'mixture-modified-color-no-skip-deep': 'UNet_flex/imagenet/0to255_RF_84x84_set_size_1232457_color_no_skip_deep_dec_64x64/',
                    # 'mixture-modified-color-large-no-skip':'UNet_flex/imagenet/0to300_RF_232x232_set_size_1232457_color_no_skips_64x64/',
                    # 'mixture-color-no-skip':'UNet_flex/imagenet/0to255_RF_128x128_set_size_1232457_color_no_skip_deep_dec_128x128/',
                     # 'mixture-color-deep-no-skip':'UNet_flex/imagenet/0to255_RF_84x84_set_size_1232457_color_no_skip_deep_dec_64x64/',
                     'mixture-color-deep-no-skip-264':'UNet_flex/imagenet/0to255_RF_264x264_set_size_1232457_color_no_skip_128x128/',
            
            
                }
        # test_sets = torch.load( '/mnt/home/zkadkhodaie/ceph/datasets/imagenet/validation_color_full_64x64_list.pt', weights_only=True) 
        test_sets =  torch.load('/mnt/home/zkadkhodaie/ceph/datasets/imagenet/test_128x128_color_list.pt', weights_only=True) 
        cond_ims =torch.concatenate(test_sets)[0::50]

    ##### 
    n_channels = test_sets[0].shape[1]
    print(len(test_sets))
    
    
    
    dist_mean = torch.concatenate(test_sets).mean(dim = (0,2,3), keepdim = True) # mean RGB values 
    print(dist_mean.mean())
    K = dist_mean.shape[2]
    

    ######################################################################################################### 

    root_path = '/mnt/home/zkadkhodaie/ceph/22_representation_in_UNet_denoiser/denoisers/'

    denoisers = {}
    
    groups = paths.keys()
    for group in groups: 
        path = root_path + paths[group]
        print('loading group ' , group )
        denoisers[group] = load_learned_model(path, print_args=True)
        
    ######################################################################################################### 

     
    n_samples = 32
    
    seed = None
    im_dim = cond_ims.shape[2]
    sigmas = torch.linspace(1,0.03, 100)
    # sigmas = torch.logspace(0,-1.4, 100)

    sig_c = None
    max_iter = 100
    n_noise=10
    seed = None
    centroids = False
    max = False
    init_ims = dist_mean * torch.ones(n_samples,n_channels,im_dim,im_dim)
    sig_str = 'linspace' + str(len(sigmas))

    samples = []


    for i in range(0, len(cond_ims), n_samples ):
        start_time = time.time()
        
        print(i)
        x_c = cond_ims[i:i+n_samples]

        sample, traj, losses = self_conditional_sampling(
                                      model=denoisers[group], 
                                      x= init_ims, 
                                      x_c=x_c , 
                                      sigmas= sigmas, 
                                      sig_c=sig_c,     
                                      seed= seed,  
                                      skip=False,
                                      n_noise=n_noise,     
                                      max_iter=max_iter, 
                                      centroids=centroids,
                                      # phi_mask = phi_mask_emphasis_spe.to(device),
                                      max=max,
                                      # mask=None ,
                                      lr = .01,
                                      return_lists = True
                                      )

        
        samples.append(sample)

        print("--- %s seconds ---" % (time.time() - start_time))


    
        torch.save(samples, '/mnt/home/zkadkhodaie/projects/22_representation_in_UNet_denoiser/results/metamer_samples_testset_'+str(im_dim)+'x'+str(im_dim)+'_'+group+'_'+data_name+ '_sig_'+sig_str+'.pt' )


    
    
if __name__ == "__main__" :
    main()
