import numpy as np
import torch 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import itertools
from tqdm import tqdm
def patch_asscalar(a):
    return a.item()

setattr(np, "asscalar", patch_asscalar)
# import colour

rgb_colors = torch.quasirandom.SobolEngine(dimension=3,scramble=True).draw(n=200)

# fig, axs = plt.subplots(10,20,figsize=(15,7))
# axs = axs.flatten()

# for i,c in enumerate(rgb_colors):
#     rectangle = patches.Rectangle((0.1, 0.1), 0.8, 0.8, facecolor=c.numpy())

#     # Add the rectangle patch to the axis
#     axs[i].add_patch(rectangle)

#     # Set limits of the plot
#     axs[i].set_xlim(0, 1)
#     axs[i].set_ylim(0, 1)
#     axs[i].axis('off')
# fig.tight_layout()

all_combinations = list(itertools.product(list(rgb_colors),list(rgb_colors)))
all_combinations_idx = list(itertools.product(range(rgb_colors.shape[0]),range(rgb_colors.shape[0])))

all_delta_e = []
for i,(c1,c2) in tqdm(enumerate(all_combinations)):
    all_delta_e.append(delta_e_cie2000(color1=convert_color(sRGBColor(*c1),LabColor),color2=convert_color(sRGBColor(*c2),LabColor)))

for i,c in enumerate(rgb_colors):
    idx = np.where(np.array(all_combinations_idx)[:,0]==i)[0]
    print("srtop")
print("stop")
