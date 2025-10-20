import numpy as np

import os
import time
import torch
import sys
sys.path.insert(0, '../../shared_code')
from model_loader_func import * 
from dataloader_func import load_nested_dataset
from dataloader_func import *
from quality_metrics_func import *
from algorithm_inv_prob import * 


############################################################################################
def interpolate_mixture_model(model, 
                          x_c1 , 
                          x_c2,
                          stop_sig =0, ### here 
                          average_phi=False,
                          init_im=None, 
                          sig_0=1, 
                          sig_L=.01, 
                          h0=.01, 
                          beta=.01, 
                          freq=0,
                          device=None, 
                          fixed_h = False,
                          max_T=None, 
                          seed=None, 
                          output_size=None, 
                          skip=True):
    
    '''
    @x_c: conditioning image of size (B,C,H,W). It could also be a list of tensors of pre-computed phi_c
    @average_phi: if True, the network uses phi_c computed from a batch of images (or pre-saved from batch of images)
    @init_im: if not None, synthesis starts from init_im rather than white noise. Size=(B,C, H,W)
    @sig_0: initial sigma (largest)
    @sig_L: final sigma (smallest)
    @h0: 1st step size
    @beta:controls added noise in each iteration (0,1]. if 1, no noise is added. As it decreases more noise added.
    @output_size: if None, same as x_c. Else, size of the generated images: (B', C, H', W'). Should match init_im size. 
    '''

    ## Set B, C, H, W from inputs
    if output_size is not None: 
        B, C, H, W = output_size
    else:  
        if init_im is not None:
            B, C, H,W = init_im.size()
        else: 
            if type(x_c) != list:
                B, C, H,W = x_c.size()
            else: 
                raise TypeError('Output dimensions is not provided')
    
    ## ambient dimensionality                 
    N = C*H*W 
    
    ## set the seed
    if seed is not None: 
        torch.manual_seed(seed)

    ## initialize the init image 
    if init_im is not None:
        y = init_im 
    else: 
        e = torch.zeros((B,C ,H,W), requires_grad= False , device=device)
        y = torch.normal(e, sig_0).to(device)
    y.requires_grad = False

    ## lists to collect stuff 
    intermed_Ys=[]
    sigmas = []
    means = []
    
    if freq > 0:
        intermed_Ys.append(y)

    t=1
    sigma = torch.ones(B)*sig_0
    start_time_total = time.time()
    x_c1 = torch.vstack([x_c1]*B ) ##### here 
    x_c2 = torch.vstack([x_c2]*B ) ##### here 
    
    update_mask = torch.ones((B,1,1,1), device=device)
    while sigma.max() > sig_L :  
        
        h = h0
        
        if fixed_h is False:
            h = h0*t/(1+ (h0*(t-1)) )

        x_c1_noisy = x_c1+ torch.randn(size = x_c1.size() , device = device) * sigma.reshape((B, 1,1,1)).to(device)
        x_c2_noisy = x_c2+ torch.randn(size = x_c2.size() , device = device) * sigma.reshape((B, 1,1,1)).to(device)
                
        with torch.no_grad():
            if sigma.mean() > stop_sig:
         
                if skip:
                    f_y = model(y, x_c1_noisy, average_phi)             
                else: 
                    f_y = y - model(y, x_c1_noisy, average_phi)             
                    
            else: 
 
                if skip:
                    f_y = model(y, x_c2_noisy, average_phi)             
                else: 
                    f_y = y - model(y, x_c2_noisy, average_phi)             
                    
                
                
        
        sigma = torch.norm(f_y, dim=(2,3),keepdim=True).norm(dim=1,keepdim=True)/np.sqrt(N)
        
        sigmas.append(sigma)
        
        gamma = sigma*np.sqrt(((1 - (beta*h))**2 - (1-h)**2 ))
        noise = torch.randn(B,C, H, W, device=device) 
        
        update_mask[sigma<sig_L] = 0 
        if freq > 0 and t%freq== 0:
            print('-----------------------------', t)
            print('sigma ' , sigma.mean().item() )
            print('mean ', y.mean().item() )
            intermed_Ys.append( (y- f_y)*update_mask )
            
        y = y -  (h*f_y + gamma*noise ) * update_mask
        means.append(y.mean(dim=(2,3)) )
        # snr = 20*torch.log10((y.std()/sigma)).item()        
        
        
        t +=1
        if max_T is not None and t>max_T:
            print('max T surpassed')
            break
        if sigma.max() > 2:
            print('not converging')
            break
    print('-------- total number of iterations: ', t)
    print("-------- final sigma, " , sigma.mean().item() )
    print('-------- final mean ', y.mean(dim=(2,3)).mean().item() )
    print("-------- final snr, " , 20*torch.log10((y.std()/sigma)).mean().item() )


    if skip:
        if stop_sig > 0 :
            denoised_y = y - model(y,x_c2_noisy, average_phi)  
        else: 
            denoised_y = y - model(y,x_c1_noisy, average_phi)  
            
    else: 
        if stop_sig > 0 :
            denoised_y = model(y,x_c2_noisy, average_phi)  
        else: 
            denoised_y = model(y,x_c1_noisy, average_phi)  
            


    return denoised_y, intermed_Ys, sigmas, means


############################################################################################

def main():
    data_name = 'six_class'
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
            fine_ids =[7,14, 20, 24, 67, 110, 132,151,199,200,246,248,271,350,374,415,439,457,520,526,560,617,626,638,659,733,742,764, 814, 817, 846, 869, 902, 937,977, 970, 1000,1015, 1040, 1069,1073,1150,1193,1261,1298,1412,1415,1455,1464,1518,1532,1551,1590, 1680, 1681]
            fine_scale_ims = torch.concatenate([train_sets[id][77:78] for id in fine_ids])
            coarse_ids = [x for x in my_ids if x not in fine_ids]
            coarse_scale_ims = torch.concatenate([train_sets[id][77:78] for id in coarse_ids])
                    
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
                     'mixture-modified-color-no-skip-deep': 'UNet_flex/imagenet/0to255_RF_84x84_set_size_1232457_color_no_skip_deep_dec_64x64/'
                    # 'mixture-modified-color-large-no-skip':'UNet_flex/imagenet/0to300_RF_232x232_set_size_1232457_color_no_skips_64x64/'
                }        
        train_sets = load_nested_dataset(folder_path = '/mnt/home/zkadkhodaie/ceph/datasets/imagenet/train/'
                        , s=(64,64) ,n_folders=200,n_images=10,crop=True, shuffle_images=False, shuffle_folders=False, 
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
            
        cond_ims= torch.vstack([train_sets[id][0:1] for id in range(len(categories))])
        coarse_scale_ims = torch.cat([train_sets[id][0:10] for id in [0,1,5]  ])
        coarse_ids = range(len(coarse_scale_ims))
        fine_scale_ims = torch.cat([train_sets[id][0:10] for id in [3,4,2] ])
        fine_ids = range(len(fine_scale_ims))
    ##### 
    print('number of images in each class:', train_sets[0].shape)
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

    start_time_total = time.time()        
    
    n_samples = 1
    
    seed = 1
    sig_L = .05
    h0 = .01
    beta = .01
    freq = 0
    fixed_h = True
    if 'skip' in group.split('-'):
        skip =False
    else: 
        skip = True
        
    all_samples = {}
    all_interm = {}
    for group in groups:
        for i , j in zip(range(len(coarse_ids)), range(len(fine_ids))):
            temp = []
            
            # for stop_sig in torch.linspace(0,1,10): 
            for stop_sig in torch.cat([torch.zeros(1),torch.logspace(-1.2,0,9)]):
                if seed is not None: 
                    torch.manual_seed(seed)
                    
                print('------------------ ',stop_sig, ' ------------------')
                if 'gray' in group.split('-'): 
                    n_channels = 1        
                    x_c1 = coarse_scale_ims[i:i+1].mean(dim = 1, keepdim = True)
                    x_c2 = fine_scale_ims[j:j+1].mean(dim = 1, keepdim = True) 
                    init_im = dist_mean.mean(dim = 0, keepdim = True).to(device) +torch.randn(n_samples,n_channels,K,K, device = device).to(device)
                else: 
                    n_channels = 3
                    x_c1 = coarse_scale_ims[i:i+1]
                    x_c2 = fine_scale_ims[j:j+1]
                    init_im = dist_mean.to(device) +torch.randn(n_samples,n_channels,K,K, device = device).to(device)
            
                       
            
            
                sample, interm_Ys,sigmas, _ = interpolate_mixture_model(denoisers[group], 
                                                                        x_c1 = x_c1.to(device), 
                                                                        x_c2 = x_c2.to(device),
                                                                        stop_sig = stop_sig,
                                                                        average_phi=False ,
                                                                        init_im =init_im,
                                                                        sig_0=1, 
                                                                        sig_L=sig_L, 
                                                                        h0=h0 , 
                                                                        beta=beta , 
                                                                        freq=freq,
                                                                        device=device, 
                                                                        fixed_h = fixed_h,
                                                                        max_T=10000, 
                                                                        seed=seed, 
                                                                        output_size=(n_samples,n_channels,K,K), 
                                                                        skip = skip)
                
            
                temp.append(sample.detach())
            all_samples[(i,j)] = torch.concatenate(temp)
            
            torch.save(all_samples, '../results/interpolation_'+group+'_'+data_name+'.pt' )
            
    
    print("--- %s seconds ---" % (time.time() - start_time_total))
    
if __name__ == "__main__" :
    main()    
