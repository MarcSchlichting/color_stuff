import pandas as pd
from colormath.color_objects import sRGBColor, HSLColor
from colormath.color_conversions import convert_color
import ast
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle
import matplotlib.colors as mcolors
import matplotlib.patches as patches

colors = [(0.8, 0.0, 0.0),  # Red at 0.0
          (1.0, 1.0, 1.0),  # Gray at 0.5
          (0.0, 0.7, 0.0)]  # Green at 1.0

# Create the colormap
cmap_name = 'custom_red_gray_green'
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors)

with open("./candidate_colors_hsl/choices3_corrected.pkl","rb") as f:
    choices = pickle.load(f)
colors_all_rgb = np.load("./candidate_colors_hsl/colors3.npy")

# with open("./candidate_colors_hsl/choices3_corrected.pkl","rb") as f:
# #     choices = pickle.load(f)
# # colors_all_rgb = np.load("./candidate_colors_hsl/colors3.npy")

bins = 12
hue_boundaries = np.linspace(0,360,bins+1)

positive_counts = np.zeros((bins,bins))
negative_counts= np.zeros((bins,bins))

for i in range(len(choices)):
    anchor_rgb = colors_all_rgb[i][0]
    A_rgb = colors_all_rgb[i][1]
    B_rgb = colors_all_rgb[i][2]
    preference = choices[i]

    anchor_hsl = convert_color(sRGBColor(*anchor_rgb),HSLColor).get_value_tuple()
    A_hsl = convert_color(sRGBColor(*A_rgb),HSLColor).get_value_tuple()
    B_hsl = convert_color(sRGBColor(*B_rgb),HSLColor).get_value_tuple()

    anchor_hue_idx = np.digitize(anchor_hsl[0],hue_boundaries) - 1  #0 bin is for values below lowest boundary
    A_hue_idx = np.digitize(A_hsl[0],hue_boundaries) - 1  #0 bin is for values below lowest boundary
    B_hue_idx = np.digitize(B_hsl[0],hue_boundaries) - 1  #0 bin is for values below lowest boundary

    if preference == "A":
        positive_counts[anchor_hue_idx,A_hue_idx] += 1
        positive_counts[A_hue_idx,anchor_hue_idx] += 1

        negative_counts[anchor_hue_idx,B_hue_idx] += 1
        negative_counts[B_hue_idx,anchor_hue_idx] += 1
    
    elif preference == "B":
        positive_counts[anchor_hue_idx,B_hue_idx] += 1
        positive_counts[B_hue_idx,anchor_hue_idx] += 1

        negative_counts[anchor_hue_idx,A_hue_idx] += 1
        negative_counts[A_hue_idx,anchor_hue_idx] += 1

# preference table
preference_table = np.zeros((bins,bins))
alpha_table = np.ones((bins,bins))
beta_table = np.ones((bins,bins))

preference_table = 2 * (positive_counts/(negative_counts+positive_counts)) - 1
alpha_table = positive_counts + 1
beta_table = negative_counts + 1


print((alpha_table+beta_table-2).min())

# plt.matshow(alpha_table+beta_table-2)
# for (i, j), z in np.ndenumerate((alpha_table+beta_table-2)):
#     plt.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

# plt.show()
hues = np.linspace(0, 1, 13)[:12] + 1/24  # Generate hues from 0 to 1
colors = [mcolors.hsv_to_rgb((hue, 0.7, 1)) for hue in hues] 
x = np.arange(12)

fig,ax = plt.subplots()
p = ax.matshow(2*(alpha_table/(alpha_table+beta_table)) - 1,vmin=-1,vmax=1,cmap=custom_cmap)
ax.set_xticks(x)
ax.set_xticklabels([])
ax.set_yticks(x)
ax.set_yticklabels([])
# Add color patches in place of x-tick labels
for i, color in enumerate(colors):
    rect = patches.Rectangle((x[i] - 0.3, -1.4), 0.6, 0.6, facecolor=color, transform=ax.transData, clip_on=False)
    ax.add_patch(rect)
    rect = patches.Rectangle((-1.4,x[i] - 0.3), 0.6, 0.6, facecolor=color, transform=ax.transData, clip_on=False)
    ax.add_patch(rect)
plt.colorbar(p)
plt.savefig("hue_analysis.pdf")
plt.show()
print("stop")