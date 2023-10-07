import numpy as np
import os
import pcutils as pcu

path = '/home/phoenix/Experiments/AK/NoDe/finalNoDe2X/'

for i in os.listdir(path):
    xyz = np.loadtxt(path+i)
    color = (0, 128, 155)
    colors = np.full(xyz.shape, color)
    pcu.polyRenderPC(xyz, colors)
