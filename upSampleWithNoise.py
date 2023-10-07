###`TP-NoDe official implementation`
### Author : Akash S Kumbar
### Final code for publishing online

import torch
import numpy as np
from noising_utils import noise
import pcutils as pcu
from denoising_utils import score_based
import argparse
import os
import math
from tqdm import tqdm

class NoDe():
    def __init__(self, xyz, mode='global', std=0.01, patch_size=256, n_neighbours=32, radius=0.2, seed_k=3):
        self.xyz = xyz
        self.mode = mode
        self.std = std
        self.patch_size = patch_size
        self.n_neighbours = n_neighbours
        self.radius = radius
        self.seed_k = seed_k
        self.min_k = 64
        self.max_k = 256

    def concatNoise(self, type='Gaussian'):
        if self.mode == 'global':
            xyz = noise.addNoiseToPC(self.xyz, self.std, type)
            noisy_patches = pcu.convertToPatchKNN(xyz, patch_size=self.patch_size, seed_k=self.seed_k)
        elif self.mode == 'KNN':
            r = 2
            patches = pcu.convertToPatchKNN(self.xyz, patch_size=self.patch_size, seed_k=self.seed_k)
            noisy_patches = torch.empty((patches.shape[0], r*patches.shape[1], 3))
            for i in range(patches.shape[0]):
                noisy_patches[i] = noise.addNoiseToPC(patches[i], self.std, type)
        elif self.mode == 'BQ': ##Ball query
            r=2 ## Upsampling factor
            patches = pcu.convertToPatchBQ(self.xyz, radius=self.radius, max_patch_size=self.patch_size, seed_k=self.seed_k)
            noisy_patches = torch.empty((patches.shape[0], r*patches.shape[1], 3))
            for i in range(patches.shape[0]):
                noisy_patches[i] = noise.addNoiseToPC(patches[i], self.std, type)
        elif self.mode == 'dilated_BQ':
            patches = pcu.convertToPatchDilatedBQ(self.xyz, base_radius=self.radius, seed_k=self.seed_k)
            noisy_patches = []
            for i in patches:
                noisy_patches.append(noise.addNoiseToPC(i, self.std, type))
        elif self.mode == 'DAKNN':
            r = 2
            patches = pcu.convertToPatchDAKNN(self.xyz, min_k = self.min_k, max_k = self.max_k)[0]
            noisy_patches = []
            for i in patches:
                noisy_patches.append(noise.addNoiseToPC(i, self.std, type))
        return noisy_patches
    
    
    def denoise(self, patches, ogNpoints, method='score_based'):
        if method == 'score_based':
            if type(patches) is not list:
                pc_denoised = score_based.denoise(ogNpoints*2, patches)
            else:
                denoised_patches = []
                for i in patches:
                    i = i.unsqueeze(0).cuda()
                    denoised_patches.append(score_based.denoise(i.shape[1]*2, i))
                pc_denoised = torch.cat(denoised_patches)  
                pc_denoised = pc_denoised.unsqueeze(0)
                # print(ogNpoints)
                pc_denoised = pcu.farthest_point_sampling(pc_denoised, ogNpoints*2)[0].squeeze(0)
        return pc_denoised
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--noising', type=str, default='KNN')
    parser.add_argument('--upsampling_factor', type=int, default=2)
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--radius', type=float, default=0.1)    
    parser.add_argument('--seed_k', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=2023)
    # parser.add_argument('--std',type=float, default=0.005)
    # Denoiser parameters
    parser.add_argument('--denoising', type=str, default='score_based')
    parser.add_argument('--noise_type', type=str, default='Gaussian')
    parser.add_argument('--data_path', type=str, default='./data/test/')
    parser.add_argument('--save_path', type=str, default='./Output/')
    args = parser.parse_args()
    assert(math.log2(args.upsampling_factor).is_integer() == True)
    its = int(math.log2(args.upsampling_factor))
    data_path = args.data_path
    save_dir = args.save_path
    device = torch.device(args.device)
    if os.path.exists(save_dir) == False:
        print(f"Creating {save_dir}.")
        os.makedirs(save_dir)

    for i in tqdm(os.listdir(data_path)):
        xyz = pcu.readOff(data_path+i, 8192)
        init_points = 2048
        # idx = np.random.choice(xyz.shape[0], init_points, replace = False)
        # xyz = xyz[idx]
        xyz = pcu.farthest_point_sampling(torch.from_numpy(xyz).unsqueeze(0), init_points)[0].squeeze(0).numpy()
        xyz, centroid, scale = pcu.normalize_point_cloud(xyz)
        xyz = torch.from_numpy(xyz).to(device)
        std = 0.005
        for j in tqdm(range(its)):
            process = NoDe(xyz, mode=args.noising, std=std, patch_size=args.patch_size, radius=args.radius, seed_k=args.seed_k)
            patches = process.concatNoise(type=args.noise_type)
            if type(patches) is not list:
                patches = patches.cuda()
            denoised = process.denoise(patches, init_points, method=args.denoising)
            std += 0.005
            init_points *= 2
            xyz = denoised
        xyz = xyz.cpu().numpy()
        xyz = xyz * scale + centroid
        color = (128, 156, 0)
        colors = np.full(xyz.shape, color)
        # print(i)
        # pcu.polyRenderPC(xyz, colors)
        np.savetxt(f'{save_dir}{i}.txt', xyz)
    
            
    # xyz = torch.randn((1024, 3)).cuda()
    # process = NoDe(xyz, mode=args.noising, std=0.01, patch_size=args.patch_size, radius=args.radius)
    # patches = process.concatNoise()
    # denoised = process.denoise(patches, 1024, method = args.denoising)
    # print(denoised.shape)
