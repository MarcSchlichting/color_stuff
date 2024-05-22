import pandas as pd
from colormath.color_objects import sRGBColor, HSLColor
from colormath.color_conversions import convert_color
import ast
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import matplotlib.patches as patches

colors = [(0.8, 0.0, 0.0),  # Red at 0.0
          (1.0, 1.0, 1.0),  # Gray at 0.5
          (0.0, 0.7, 0.0)]  # Green at 1.0

# Create the colormap
cmap_name = 'custom_red_gray_green'
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors)

df = pd.read_csv("preferences.csv")

bins = 10
lightness_boundaries = np.linspace(0,1,bins+1)

counts_empty = [{"to_bin_idx":i, "to_bin_boundaries":[lightness_boundaries[i],lightness_boundaries[i+1]],"positive":0,"negative":0} for i in range(bins)]
counts = [{"from_bin_idx":i, "from_bin_boundaries":[lightness_boundaries[i],lightness_boundaries[i+1]], "to_counts":deepcopy(counts_empty)} for i in range(bins)]

for i in range(df.shape[0]):
    anchor_rgb = ast.literal_eval(df["Anchor_RGB"].iloc[i])
    A_rgb = ast.literal_eval(df["Color_A_RGB"].iloc[i])
    B_rgb = ast.literal_eval(df["Color_B_RGB"].iloc[i])
    preference = df["Preference"].iloc[i]

    anchor_hsl = convert_color(sRGBColor(*anchor_rgb),HSLColor).get_value_tuple()
    A_hsl = convert_color(sRGBColor(*A_rgb),HSLColor).get_value_tuple()
    B_hsl = convert_color(sRGBColor(*B_rgb),HSLColor).get_value_tuple()

    anchor_lightness_idx = np.digitize(anchor_hsl[2],lightness_boundaries) - 1  #0 bin is for values below lowest boundary
    A_lightness_idx = np.digitize(A_hsl[2],lightness_boundaries) - 1  #0 bin is for values below lowest boundary
    B_lightness_idx = np.digitize(B_hsl[2],lightness_boundaries) - 1  #0 bin is for values below lowest boundary

    for from_idx in range(bins):
        # case 1: anchor is in bin
        if anchor_lightness_idx == from_idx:
            if preference == "A":
                # positive
                counts[from_idx]["to_counts"][A_lightness_idx]["positive"] += 1
                #negative
                counts[from_idx]["to_counts"][B_lightness_idx]["negative"] += 1

            elif preference == "B":
                # positive
                counts[from_idx]["to_counts"][B_lightness_idx]["positive"] += 1
                #negative
                counts[from_idx]["to_counts"][A_lightness_idx]["negative"] += 1
            
            else:
                raise ValueError("Preference should either be 'A' or 'B'.")
        
        else:
            if A_lightness_idx == from_idx:
                if preference == "A":
                    counts[from_idx]["to_counts"][anchor_lightness_idx]["positive"] += 1

                elif preference == "B":
                    counts[from_idx]["to_counts"][anchor_lightness_idx]["negative"] += 1

                else:
                    raise ValueError("Preference should either be 'A' or 'B'.")
                
            if B_lightness_idx == from_idx:
                if preference == "B":
                    counts[from_idx]["to_counts"][anchor_lightness_idx]["positive"] += 1

                elif preference == "A":
                    counts[from_idx]["to_counts"][anchor_lightness_idx]["negative"] += 1

                else:
                    raise ValueError("Preference should either be 'A' or 'B'.")

# preference table
preference_table = np.zeros((bins,bins))
alpha_table = np.ones((bins,bins))
beta_table = np.ones((bins,bins))

for from_idx in range(bins):
    for to_idx in range(bins):
        positive_counts = counts[from_idx]["to_counts"][to_idx]["positive"]
        negative_counts = counts[from_idx]["to_counts"][to_idx]["negative"]

        if positive_counts+negative_counts == 0:
            preference_table[from_idx,to_idx] = 0
        else:
            preference_table[from_idx,to_idx] = 2*(positive_counts/(negative_counts+positive_counts)) - 1   
        alpha_table[from_idx,to_idx] += positive_counts
        beta_table[from_idx,to_idx] += negative_counts

print((alpha_table+beta_table-2).min())

plt.matshow(alpha_table+beta_table-2)
for (i, j), z in np.ndenumerate((alpha_table+beta_table-2)):
    plt.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

plt.show()

luminosities = np.linspace(0.05, 0.95, 10)  # Generate hues from 0 to 1
colors = [convert_color(HSLColor(220,0.7,lum),sRGBColor).get_value_tuple() for lum in luminosities] 
x = np.arange(10)

fig,ax = plt.subplots()
# p = ax.matshow(preference_table,vmin=-1,vmax=1,cmap=custom_cmap)
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
plt.savefig("luminosity_analysis.pdf")
plt.show()
print("stop")