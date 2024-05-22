import numpy as np
import matplotlib.pyplot as plt
from colormath.color_objects import sRGBColor, LabColor, HSLColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from scipy.optimize import minimize_scalar
from tqdm import tqdm
import os
from joblib import Parallel, delayed
import itertools

def patch_asscalar(a):
    return a.item()

setattr(np, "asscalar", patch_asscalar)


def evaluate_scaling(scale,anchor,direction,desired_delta_e):
    candidate = anchor + scale * direction
    return np.abs(desired_delta_e - delta_e_cie2000(LabColor(*anchor),LabColor(*candidate)))

def find_color_with_delta_e(anchor,direction,desired_delta_e):
    res = minimize_scalar(evaluate_scaling,bounds=(0,120),args=(anchor,direction,desired_delta_e))
    return anchor + res.x * direction, res.success


def get_color_sample(delta_e=20,diff_c_min=1):
    anchor_rgb = np.random.rand(3,)
    anchor_lab = np.array(convert_color(sRGBColor(*anchor_rgb),LabColor).get_value_tuple())
    found_candidate1 = False
    found_candidate2 = False
    
    counter = 0
    while not found_candidate1:    
        direction1 = np.random.rand(3,)
        direction1 /= np.linalg.norm(direction1)
        candidate1_lab, success = find_color_with_delta_e(anchor_lab,direction1,delta_e)
        success
        if not success:
            continue
        candidate1_rgb = np.array(convert_color(LabColor(*candidate1_lab),sRGBColor).get_value_tuple())
        if np.all((candidate1_rgb >= 0) & (candidate1_rgb <= 1)):
            found_candidate1 = True
        counter += 1
        if counter >= 1000:
            raise ValueError
    
    counter = 0
    while not found_candidate2:    
        direction2 = np.random.rand(3,)
        direction2 /= np.linalg.norm(direction2)
        candidate2_lab, success = find_color_with_delta_e(anchor_lab,direction2,delta_e)
        if not success:
            continue
        candidate2_rgb = np.array(convert_color(LabColor(*candidate2_lab),sRGBColor).get_value_tuple())
        if np.all((candidate2_rgb >= 0) & (candidate2_rgb <= 1)) & (delta_e_cie2000(LabColor(*candidate1_lab),LabColor(*candidate2_lab)) >= diff_c_min):
            found_candidate2 = True
        counter += 1
        if counter >= 1000:
            raise ValueError
    
    return anchor_rgb, candidate1_rgb, candidate2_rgb    

# generate sample data
n_points = 201
x = np.linspace(0,10,n_points)
y1 = np.exp(-x) + 0.05 * np.random.randn(n_points)
y2 = 2/(1+np.exp(x-3))  + 0.05 * np.random.randn(n_points)


colors = []
hue_values = []
hues = np.arange(0,360,30)
# combinations_c1_c2 = list(itertools.combinations(hues,2))
combinations_all = list(itertools.combinations(hues,3))
sat = 0.7
light = 0.5
for anch_hue,col1_hue,col2_hue in combinations_all:
    anchor = convert_color(HSLColor(anch_hue,sat,light),sRGBColor).get_value_tuple()
    c1 = convert_color(HSLColor(col1_hue,sat,light),sRGBColor).get_value_tuple()
    c2 = convert_color(HSLColor(col2_hue,sat,light),sRGBColor).get_value_tuple()

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
    fig.suptitle(f"{len(colors)}")
    fig.tight_layout()
    fig.savefig(f"candidate_colors_hsl/{len(colors)}.pdf")
    plt.close()

    colors.append((anchor,c1,c2))
    hue_values.append((anch_hue,col1_hue,col2_hue))
    print(len(colors))

colors = np.stack(colors)
hues_values = np.stack(hue_values)

np.save(f"candidate_colors_hsl/colors.npy",colors)
np.save(f"candidate_colors_hsl/hues.npy",colors)
        




# plt.show()