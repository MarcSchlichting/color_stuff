import pandas as pd
from colormath.color_objects import sRGBColor, HSLColor
from colormath.color_conversions import convert_color
import ast
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle
from dkregression.kernels import RBF
from dkregression.likelihoods import BernoulliLikelihood
from dkregression.cross_validation import CrossValidation
from dkregression import DKR
import torch
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

# responses from the online survey
df = pd.read_csv("./preferences.csv")
choices = list(df["Preference"])

colors_all_rgb = []
for i in range(df.shape[0]):
    colors_all_rgb.append(np.stack([eval(f"np.array({df[col].iloc[i]})") for col in ["Anchor_RGB","Color_A_RGB","Color_B_RGB"]],axis=0))
colors_all_rgb = np.stack(colors_all_rgb,axis=0)


positive_hsl_combinations = []
negative_hsl_combinations = []

for i in range(len(choices)):
    anchor_rgb = colors_all_rgb[i][0]
    A_rgb = colors_all_rgb[i][1]
    B_rgb = colors_all_rgb[i][2]
    preference = choices[i]

    anchor_hsl = convert_color(sRGBColor(*anchor_rgb),HSLColor).get_value_tuple()
    A_hsl = convert_color(sRGBColor(*A_rgb),HSLColor).get_value_tuple()
    B_hsl = convert_color(sRGBColor(*B_rgb),HSLColor).get_value_tuple()


    if preference == "A":
        positive_hsl_combinations.append(torch.Tensor(anchor_hsl+A_hsl))
        positive_hsl_combinations.append(torch.Tensor(A_hsl+anchor_hsl))
        
        negative_hsl_combinations.append(torch.Tensor(anchor_hsl+B_hsl))
        negative_hsl_combinations.append(torch.Tensor(B_hsl+anchor_hsl))
    
    elif preference == "B":
        negative_hsl_combinations.append(torch.Tensor(anchor_hsl+A_hsl))
        negative_hsl_combinations.append(torch.Tensor(A_hsl+anchor_hsl))
        
        positive_hsl_combinations.append(torch.Tensor(anchor_hsl+B_hsl))
        positive_hsl_combinations.append(torch.Tensor(B_hsl+anchor_hsl))

positive_hue_combinations = torch.stack(positive_hsl_combinations,dim=0)
negative_hue_combinations = torch.stack(negative_hsl_combinations,dim=0)


X = torch.concatenate([positive_hue_combinations,negative_hue_combinations],dim=0) / torch.Tensor([[360,1,1,360,1,1]])
y = torch.concatenate([torch.ones((positive_hue_combinations.shape[0],1)),torch.zeros((negative_hue_combinations.shape[0],1))],dim=0)


#Fitting
kernel = RBF(X)
likelihood = BernoulliLikelihood()
cv = CrossValidation()
model = DKR(kernel,likelihood,cv)
model.fit(X,y,verbose=True)

xq1,xq2 = torch.meshgrid(torch.linspace(0,1,101),torch.linspace(0,1,101))
query_coordinates = torch.stack([xq1,xq2],dim=-1).reshape(-1,2)

xq = torch.ones((query_coordinates.shape[0],6))


# predicted_preferences = model.predict(query_coordinates)
# predicted_preferences_mesh = {k:predicted_preferences[k].reshape(xq1.shape) for k in predicted_preferences}

# x = np.linspace(1/24,1-1/24,12)
# colors = [mcolors.hsv_to_rgb((hue, 0.7, 1)) for hue in x]

# fig,ax = plt.subplots(1,3,figsize=(20,6))

# #Hue
# p = ax[0].contourf(xq1.numpy(),xq2.numpy(),2*(predicted_preferences_mesh["p"].numpy())-1,levels=20,vmin=-1.0,vmax=1.0,origin="upper",cmap=custom_cmap)
# ax[0].scatter(*X.T,s=2,color="black",alpha=0.2)
# ax[0].set_xticks(x)
# ax[0].set_xticklabels([])
# ax[0].set_yticks(x)
# ax[0].set_yticklabels([])
# ax[0].set_title("Hue")
# # Add color patches in place of x-tick labels
# for i, color in enumerate(colors):
#     rect = patches.Rectangle((x[i]-0.03,-0.08), 0.06, 0.06, facecolor=color, transform=ax[0].transData, clip_on=False)
#     ax[0].add_patch(rect)
#     rect = patches.Rectangle((-0.08,x[i]-0.03), 0.06, 0.06, facecolor=color, transform=ax[0].transData, clip_on=False)
#     ax[0].add_patch(rect)
# plt.colorbar(p)
# # plt.gca().invert_yaxis()
# # ax[0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
# # plt.xlabel("Hue 1 [deg]")
# # plt.ylabel("Hue 2 [deg]")
# # plt.tight_layout()
# # plt.savefig("hue_analysis_dkr.pdf")
# # plt.show()




# # Saturation

# # responses from the online survey
# df = pd.read_csv("./preferences.csv")
# choices = list(df["Preference"])

# colors_all_rgb = []
# for i in range(df.shape[0]):
#     colors_all_rgb.append(np.stack([eval(f"np.array({df[col].iloc[i]})") for col in ["Anchor_RGB","Color_A_RGB","Color_B_RGB"]],axis=0))
# colors_all_rgb = np.stack(colors_all_rgb,axis=0)

# positive_sat_combinations = []
# negative_sat_combinations = []

# for i in range(len(choices)):
#     anchor_rgb = colors_all_rgb[i][0]
#     A_rgb = colors_all_rgb[i][1]
#     B_rgb = colors_all_rgb[i][2]
#     preference = choices[i]

#     anchor_hsl = convert_color(sRGBColor(*anchor_rgb),HSLColor).get_value_tuple()
#     A_hsl = convert_color(sRGBColor(*A_rgb),HSLColor).get_value_tuple()
#     B_hsl = convert_color(sRGBColor(*B_rgb),HSLColor).get_value_tuple()

#     idx = 1

#     if preference == "A":
#         positive_sat_combinations.append((anchor_hsl[idx],A_hsl[idx]))
#         positive_sat_combinations.append((A_hsl[idx],anchor_hsl[idx]))
        
#         negative_sat_combinations.append((anchor_hsl[idx],B_hsl[idx]))
#         negative_sat_combinations.append((B_hsl[idx],anchor_hsl[idx]))
    
#     elif preference == "B":
#         negative_sat_combinations.append((anchor_hsl[idx],A_hsl[idx]))
#         negative_sat_combinations.append((A_hsl[idx],anchor_hsl[idx]))
        
#         positive_sat_combinations.append((anchor_hsl[idx],B_hsl[idx]))
#         positive_sat_combinations.append((B_hsl[idx],anchor_hsl[idx]))

# positive_sat_combinations = np.array(positive_sat_combinations)
# negative_sat_combinations = np.array(negative_sat_combinations)

# X = torch.concatenate([torch.Tensor(positive_sat_combinations),torch.Tensor(negative_sat_combinations)],dim=0)  # no normalization as already 0...1
# y = torch.concatenate([torch.ones((positive_sat_combinations.shape[0],1)),torch.zeros((negative_sat_combinations.shape[0],1))],dim=0)

# kernel = RBF(X)
# likelihood = BernoulliLikelihood()
# cv = CrossValidation()
# model = DKR(kernel,likelihood,cv)
# model.fit(X,y,verbose=True)

# xq1,xq2 = torch.meshgrid(torch.linspace(0,1,101),torch.linspace(0,1,101))
# query_coordinates = torch.stack([xq1,xq2],dim=-1).reshape(-1,2)


# predicted_preferences = model.predict(query_coordinates)
# predicted_preferences_mesh = {k:predicted_preferences[k].reshape(xq1.shape) for k in predicted_preferences}

# x = np.linspace(1/24,1-1/24,12)
# colors = [convert_color(HSLColor(220,sat,0.5),sRGBColor).get_value_tuple() for sat in x]

# p = ax[1].contourf(xq1.numpy(),xq2.numpy(),2*(predicted_preferences_mesh["p"].numpy())-1,levels=20,vmin=-1.0,vmax=1.0,origin="upper",cmap=custom_cmap)
# ax[1].scatter(*X.T,s=2,color="black",alpha=0.2)
# ax[1].set_xticks(x)
# ax[1].set_xticklabels([])
# ax[1].set_yticks(x)
# ax[1].set_yticklabels([])
# ax[1].set_title("Saturation")
# # Add color patches in place of x-tick labels
# for i, color in enumerate(colors):
#     rect = patches.Rectangle((x[i]-0.03,-0.08), 0.06, 0.06, facecolor=color, transform=ax[1].transData, clip_on=False)
#     ax[1].add_patch(rect)
#     rect = patches.Rectangle((-0.08,x[i]-0.03), 0.06, 0.06, facecolor=color, transform=ax[1].transData, clip_on=False)
#     ax[1].add_patch(rect)
# plt.colorbar(p)
# plt.gca().invert_yaxis()
# # ax[1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)


# # Lightness


# positive_light_combinations = []
# negative_light_combinations = []

# for i in range(len(choices)):
#     anchor_rgb = colors_all_rgb[i][0]
#     A_rgb = colors_all_rgb[i][1]
#     B_rgb = colors_all_rgb[i][2]
#     preference = choices[i]

#     anchor_hsl = convert_color(sRGBColor(*anchor_rgb),HSLColor).get_value_tuple()
#     A_hsl = convert_color(sRGBColor(*A_rgb),HSLColor).get_value_tuple()
#     B_hsl = convert_color(sRGBColor(*B_rgb),HSLColor).get_value_tuple()

#     idx = 2

#     if preference == "A":
#         positive_light_combinations.append((anchor_hsl[idx],A_hsl[idx]))
#         positive_light_combinations.append((A_hsl[idx],anchor_hsl[idx]))
        
#         negative_light_combinations.append((anchor_hsl[idx],B_hsl[idx]))
#         negative_light_combinations.append((B_hsl[idx],anchor_hsl[idx]))
    
#     elif preference == "B":
#         negative_light_combinations.append((anchor_hsl[idx],A_hsl[idx]))
#         negative_light_combinations.append((A_hsl[idx],anchor_hsl[idx]))
        
#         positive_light_combinations.append((anchor_hsl[idx],B_hsl[idx]))
#         positive_light_combinations.append((B_hsl[idx],anchor_hsl[idx]))

# positive_light_combinations = np.array(positive_light_combinations)
# negative_light_combinations = np.array(negative_light_combinations)

# X = torch.concatenate([torch.Tensor(positive_light_combinations),torch.Tensor(negative_light_combinations)],dim=0)  # no normalization as already 0...1
# y = torch.concatenate([torch.ones((positive_light_combinations.shape[0],1)),torch.zeros((negative_light_combinations.shape[0],1))],dim=0)

# kernel = RBF(X)
# likelihood = BernoulliLikelihood()
# cv = CrossValidation()
# model = DKR(kernel,likelihood,cv)
# model.fit(X,y,verbose=True)

# xq1,xq2 = torch.meshgrid(torch.linspace(0,1,101),torch.linspace(0,1,101))
# query_coordinates = torch.stack([xq1,xq2],dim=-1).reshape(-1,2)


# predicted_preferences = model.predict(query_coordinates)
# predicted_preferences_mesh = {k:predicted_preferences[k].reshape(xq1.shape) for k in predicted_preferences}

# x = np.linspace(1/24,1-1/24,12)
# colors = [convert_color(HSLColor(220,0.7,light),sRGBColor).get_value_tuple() for light in x]

# p = ax[2].contourf(xq1.numpy(),xq2.numpy(),2*(predicted_preferences_mesh["p"].numpy())-1,levels=20,vmin=-1.0,vmax=1.0,origin="upper",cmap=custom_cmap)
# ax[2].scatter(*X.T,s=2,color="black",alpha=0.2)
# ax[2].set_xticks(x)
# ax[2].set_xticklabels([])
# ax[2].set_yticks(x)
# ax[2].set_yticklabels([])
# ax[2].set_title("Lightness")
# # Add color patches in place of x-tick labels
# for i, color in enumerate(colors):
#     rect = patches.Rectangle((x[i]-0.03,-0.08), 0.06, 0.06, facecolor=color, transform=ax[2].transData, clip_on=False)
#     ax[2].add_patch(rect)
#     rect = patches.Rectangle((-0.08,x[i]-0.03), 0.06, 0.06, facecolor=color, transform=ax[2].transData, clip_on=False)
#     ax[2].add_patch(rect)
# plt.colorbar(p)
# plt.gca().invert_yaxis()
# # ax[1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)


# plt.tight_layout()
# plt.savefig("all_dkr_analysis_online.pdf")
# plt.show()
# print("stop")
