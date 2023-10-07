import torch
from score_denoise.utils.misc import *
from score_denoise.utils.denoise import *
from score_denoise.models.denoise import *


def denoise(ogPoints, patches, ld_step_size=0.2, ld_num_steps=30, patch_size=256, seed_k=3, denoise_knn=4, step_decay=0.95, get_traj=False):
    device = torch.device('cuda')
    ckpt = torch.load('score_denoise/pretrained/ckpt.pt', map_location=device)
    model = DenoiseNet(ckpt['args']).to(device)
    model.load_state_dict(ckpt['state_dict'])

    with torch.no_grad():
        model.eval()
        patches = patches.to(torch.float32)
        patches_denoised, traj = model.denoise_langevin_dynamics(patches, step_size=ld_step_size, denoise_knn=denoise_knn, step_decay=step_decay, num_steps=ld_num_steps)

    pcl_denoised, fps_idx = farthest_point_sampling(patches_denoised.view(1, -1, 3), ogPoints)
    pcl_denoised = pcl_denoised[0]
    fps_idx = fps_idx[0]

    return pcl_denoised