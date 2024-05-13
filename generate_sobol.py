import numpy as np
import torch 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import itertools
from tqdm import tqdm
import os
import pickle
def patch_asscalar(a):
    return a.item()

setattr(np, "asscalar", patch_asscalar)
# import colour

N = 200
N_elite = 5

for de in [25]:

    try:
        os.makedirs(f"candidate_colors/{de}")
    except FileExistsError:
        pass
    
    rgb_colors = torch.quasirandom.SobolEngine(dimension=3,scramble=True).draw(n=N)

    all_combinations = list(itertools.product(list(rgb_colors),list(rgb_colors)))
    all_combinations_idx = list(itertools.product(range(rgb_colors.shape[0]),range(rgb_colors.shape[0])))

    all_delta_e = []
    for i,(c1,c2) in tqdm(enumerate(all_combinations)):
        all_delta_e.append(delta_e_cie2000(color1=convert_color(sRGBColor(*c1),LabColor),color2=convert_color(sRGBColor(*c2),LabColor)))
    all_delta_e = np.array(all_delta_e).reshape(N,N)

    color_tuples = []
    for i,c in enumerate(rgb_colors):
        sub_delta_e = all_delta_e[i]
        delta_delta_e = np.abs(de - all_delta_e)
        delta_delta_idx_sorted = np.argsort(delta_delta_e[i])
        elite_idx = delta_delta_idx_sorted[:N_elite]
        elite_delta_es = all_delta_e[elite_idx][:,elite_idx]
        max_index = np.argmax(elite_delta_es)
        row_index, col_index = np.unravel_index(max_index, elite_delta_es.shape)
        c1 = rgb_colors[elite_idx[row_index]].numpy()
        c2 = rgb_colors[elite_idx[col_index]].numpy()
        color_tuples.append({'anchor':c.numpy(),"c1":c1,"c2":c2})
        print(sub_delta_e[elite_idx[row_index]],sub_delta_e[elite_idx[row_index]])

    # generate sample data
    n_points = 201
    x = np.linspace(0,10,n_points)
    y1 = np.exp(-x) + 0.05 * np.random.randn(n_points)
    y2 = 2/(1+np.exp(x-3))  + 0.05 * np.random.randn(n_points)

    for i in range(len(color_tuples)):
        anchor = color_tuples[i]['anchor']
        c1 = color_tuples[i]['c1']
        c2 = color_tuples[i]['c2']
        fig,axs = plt.subplots(1,2,figsize = (8,3))
        axs[0].plot(x,y1,c=anchor)
        axs[0].plot(x,y2,c=c1)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_title("A")
        axs[1].plot(x,y1,c=anchor)
        axs[1].plot(x,y2,c=c2)
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].set_title("B")
        fig.tight_layout()
        fig.savefig(f"candidate_colors/{de}/{i}.png",dpi=450)
        plt.close()

    # save the color tuples
    with open(f"candidate_colors/{de}/colors_{de}.pkl","wb") as f:
        pickle.dump(color_tuples,f)
print("stop")
