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
    



    ######################################################################################################### 
    ### texture 
    if data_name == 'texture':
        paths = { 
                    # 'mixture-modified-color': 'UNet_flex/texture_EPS_color/0to255_RF_84x84_set_size_237580_color_80x80/' , 
                    'mixture-modified-color-no-skip-deep': 'UNet_flex/texture_EPS_color/0to255_RF_84x84_set_size_237580_color_no_skip_deep_enc_80x80/' , 
                }
        
        data = torch.load(  '/mnt/home/zkadkhodaie/ceph/datasets/texture_EPS/patched_1024x1024_to_80x80_all_sets_color.pt',weights_only=True)
        dim = data.shape[2]
        n_patches = int(1024/dim)**2 # 144
        
        num_classes = int(data.shape[0]/n_patches) # for 80x80: 882 + 786 #for 128x128:786
        
        ##### list of tensors. Each tensor contains images from same class 
        train_sets = []
        test_sets = []
    
        if dim==80: 
            n_test = 4
        if dim ==128: 
            n_test = 1
        
        for d in range(num_classes): 
            train_sets.append(data[n_patches * d: (n_patches * (d+1)) -n_test ])
            test_sets.append(data[(n_patches * (d+1)) -n_test : (n_patches * (d+1)) ] ) 

        if dim == 80:
            my_ids = [4,7, 14,16, 20,24, 37, 41,67, 74, 96, 110, 121, 132, 144,151, 192, 199, 200,
                      233,246,248, 260, 271,343, 345, 350, 366, 374, 413, 415, 416, 439, 457, 
                     460, 492,516,520, 526,560, 597,617, 626,628, 629, 638,652,659, 665,682,
                     691, 697, 722,733,735,742, 753,754, 763, 764, 765, 778, 786, 802,805, 806, 814, 817,828, 
                     831, 843, 846,869,872, 902, 934,937,940,946, 974,977, 970,983, 1000 , 1015, 1040,1069,
                     1073,1077,1109, 1150, 1181,1193, 1261, 1298, 1321, 1412, 1415, 1455,1461,1464,1478, 1518,1532, 
                    1551,1577,1590, 1629, 1680, 1681]
            cond_ims =torch.concatenate([train_sets[id][77:78] for id in my_ids])
            
        
        elif dim == 128:
            my_ids = [ 1,2,22,56,60,77,80,85,135,145,172,175,192,210,227,242,246, 250,255,265,258,
                      270, 290, 293,309,319,326,333,334,339,342,358,367,403,416,420,425,429,431,
                      436,440,445,449,452,459,467,477,488,527,528,552,553,560,566,574,581,583,590,
                      594,599,600,602,619,635,651,652,671,705,722,725,755,776,781,786,800,809,816,835,848,
                      873,885,909,927,961,968,1015,1020,1041,1049,1066,1074,1090,1098,1099,1103,1126,1136,1268,
                      1279,1424,1431,1479,1483,1512,1550,1551,1629,1648,1666,1671] 
            cond_ims =torch.concatenate([train_sets[id][36:37] for id in my_ids])

    ### imagenet 
    elif data_name == 'imagenet':
        paths = { 
                     # 'mixture-modified-color-no-skip-deep': 'UNet_flex/imagenet/0to255_RF_84x84_set_size_1232457_color_no_skip_deep_dec_64x64/',
                    # 'mixture-modified-color-large-no-skip':'UNet_flex/imagenet/0to300_RF_232x232_set_size_1232457_color_no_skips_64x64/',
                    # 'mixture-color-no-skip':'UNet_flex/imagenet/0to255_RF_128x128_set_size_1232457_color_no_skip_deep_dec_128x128/',
                    'mixture-color-no-skip-smaller':'UNet_flex/imagenet/0to255_RF_128x128_set_size_1232457_color_no_skip_128x128/',
            
            
                }
        
        train_sets = load_nested_dataset(folder_path = '/mnt/home/zkadkhodaie/ceph/datasets/imagenet/train/'
                        , s=(128,128) ,n_folders=200,n_images=10,crop=True, shuffle_images=False, shuffle_folders=False, 
                              one_tensor=False)
    
        my_ids= [3,7,8,9,12,15,19,21,24,27,30, 38,41,59, 72, 84,92,101, 111,117, 137,157,166,167,194]
        cond_ims = torch.concatenate([train_sets[id][0:1] for id in my_ids])
        
    #### six_class
    elif data_name == 'six_class': 
        paths = { 
                  'mixture-modified-color-large' : 'UNet_flex/multi_class_dataset/0to255_RF_84x84_set_size_287946_color_80x80/',          
                }
    
        train_sets = []
        categories = ['celebaHQ', 'bedroom','nabirds','afhq', 'stanford_car', 'flowers102']
        for category in categories:
            data = torch.load( '/mnt/home/zkadkhodaie/ceph/datasets/' + category +'/train_color_80x80.pt', weights_only=True)    
            print(category, data.shape)
            N = 100
            train = data[0:N]                        
            train_sets.append(train)
            
        cond_ims= torch.vstack([train_sets[id][0:10] for id in range(len(categories))])
        my_ids = range(len(cond_ims))
    ##### 
    n_channels = train_sets[0].shape[1]
    print(len(train_sets))
    
    
    
    dist_mean = torch.concatenate(train_sets).mean(0)
    print(dist_mean.mean())
    K = dist_mean.shape[2]
    

    ######################################################################################################### 

    root_path = '/mnt/home/zkadkhodaie/ceph/21_hierarchical_conditional_prior/denoisers/'

    denoisers = {}
    
    groups = paths.keys()
    for group in groups: 
        path = root_path + paths[group]
    
        print('loading group ' , group )
        denoisers[group] = load_learned_model(path, print_args=True, new_arch = 'UNet_conditional_mean_matching')
        start_time_total = time.time()        
        print("--- %s seconds ---" % (round(time.time() - start_time_total)))
        for b in range(len(denoisers[group].matching_layers_enc)):
            denoisers[group].matching_layers_enc[b] = False
        for b in range(len(denoisers[group].matching_layers_dec)):
           denoisers[group].matching_layers_dec[b] = True
        denoisers[group].matching_layers_mid = True
        
        denoisers[group].matching_mode = 'additive'
        denoisers[group].match_std = True
        
    
    ######################################################################################################### 

    # K = 320

    
    fixed_h = True
    sig_L=.05
    h0=.01
    beta=0.01
    n_samples =10
    seed = 0
    freq =0
    #########################

    
    all_samples = {}
    
    for group in groups: 
        start_time_total = time.time()        
        if 'skip' in group.split('-'): 
            skip = False
        else: 
            skip = True
            
        im_n = 0
        for i in range(len(my_ids)):  
            id = my_ids[i]
            print('------------ class id : ', id )
            temp = []
            for _ in range(1): 
                    
                if seed is not None:
                    torch.manual_seed(seed )
                if 'color' in group.split('-'):
                    n_channels=3                 
                    x_c = cond_ims[i:i+1]
                    # init_im = dist_mean.to(device).mean() +torch.randn(n_samples,n_channels,K,K, device = device).to(device)
                    init_im = dist_mean.to(device).mean(dim = (1,2), keepdim = True) +torch.randn(n_samples,n_channels,K,K, device = device).to(device)

                else: 
                    n_channels=1
                    x_c = cond_ims[i:i+1].mean(dim = 1, keepdim = True)      
                    init_im = dist_mean.mean(dim=0, keepdim=True).to(device).mean()  +torch.randn(n_samples,n_channels,K,K, device = device).to(device)
                    
                sample, interm_Ys,_, _ = batch_conditional_synthesis(denoisers[group], 
                                                                        x_c = x_c.to(device), 
                                                                        noisy_conditioner=True,
                                                                        average_phi=True ,
                                                                        init_im =init_im,
                                                                        sig_0=1, 
                                                                        sig_L=sig_L, 
                                                                        h0=h0 , 
                                                                        beta=beta , 
                                                                        freq=freq,
                                                                        device=device, 
                                                                        fixed_h = fixed_h,
                                                                        max_T=15000, 
                                                                        seed=seed, 
                                                                        output_size=(n_samples,n_channels,K,K), 
                                                                        skip= skip)
            
                temp.append(sample.detach())
                
                    
            all_samples[id] = torch.cat(temp)      
            torch.save(all_samples, '/mnt/home/zkadkhodaie/projects/21_hierarchical_conditional_prior/results/samples_'+str(K)+'x'+str(K)+'_match_dec_mean_var_'+denoisers[group].matching_mode+'_'+group+'_'+data_name+ '.pt' )
    print("--- %s seconds ---" % (time.time() - start_time_total))
    
if __name__ == "__main__" :
    main()
