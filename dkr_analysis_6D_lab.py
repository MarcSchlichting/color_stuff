import pandas as pd
from colormath.color_objects import sRGBColor, LabColor, HSLColor
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


    anchor_hsl = convert_color(sRGBColor(*anchor_rgb),LabColor).get_value_tuple()
    A_hsl = convert_color(sRGBColor(*A_rgb),LabColor).get_value_tuple()
    B_hsl = convert_color(sRGBColor(*B_rgb),LabColor).get_value_tuple()

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


# X = torch.concatenate([positive_hue_combinations,negative_hue_combinations],dim=0) / torch.Tensor([[100,132,360,100,132,360]])
X = torch.concatenate([positive_hue_combinations,negative_hue_combinations],dim=0) / torch.Tensor([[100,128,128,100,128,128]]) + torch.Tensor([[0,1,1,0,1,1]])
y = torch.concatenate([torch.ones((positive_hue_combinations.shape[0],1)),torch.zeros((negative_hue_combinations.shape[0],1))],dim=0)


#Fitting
kernel = RBF(X)
likelihood = BernoulliLikelihood()
cv = CrossValidation()
model = DKR(kernel,likelihood,cv)
model.fit(X,y,verbose=True)

xq1,xq2 = torch.meshgrid(torch.linspace(0,1,101),torch.linspace(0,1,101))
query_coordinates = torch.stack([xq1,xq2],dim=-1).reshape(-1,2)

fig,ax = plt.subplots(5,5,figsize=(15,15))

for i,s in enumerate([0.1,0.3,0.5,0.7,0.9]):
    for j,l in enumerate([0.1,0.3,0.5,0.7,0.9]):

        # xq = torch.ones((query_coordinates.shape[0],6))
        # xq[:,0] = l
        # xq[:,1] = s
        # xq[:,2] = query_coordinates[:,0]
        # xq[:,3] = l
        # xq[:,4] = s
        # xq[:,5] = query_coordinates[:,1]
        lab1s = torch.Tensor([convert_color(HSLColor(query_coordinates[i,0]*360,s,l),LabColor).get_value_tuple() for i in range(query_coordinates[:,0].shape[0])])
        lab2s = torch.Tensor([convert_color(HSLColor(query_coordinates[i,1]*360,s,l),LabColor).get_value_tuple() for i in range(query_coordinates[:,1].shape[0])])
        
        # xq = torch.ones((query_coordinates.shape[0],6))
        # xq[:,0] = query_coordinates[:,0]
        # xq[:,1] = s
        # xq[:,2] = l
        # xq[:,3] = query_coordinates[:,1]
        # xq[:,4] = s
        # xq[:,5] = l
        xq = torch.concatenate([lab1s,lab2s],dim=-1) / torch.Tensor([[100,128,128,100,128,128]]) + torch.Tensor([[0,1,1,0,1,1]])


        predicted_preferences = model.predict(xq)
        predicted_preferences_mesh = {k:predicted_preferences[k].reshape(xq1.shape) for k in predicted_preferences}

        x = np.linspace(1/24,1-1/24,12)
        # colors = [np.clip(convert_color(LCHabColor(l*100, s*132, hue*360),sRGBColor).get_value_tuple(),0,1) for hue in x]
        colors = [np.clip(convert_color(HSLColor(hue*360,s,l),sRGBColor).get_value_tuple(),0,1) for hue in x]

        #Hue
        p = ax[i,j].contourf(xq1.numpy(),xq2.numpy(),2*(predicted_preferences_mesh["p"].numpy())-1,levels=20,vmin=-1.0,vmax=1.0,origin="upper",cmap=custom_cmap)
        ax[i,j].set_xticks(x)
        ax[i,j].set_xticklabels([])
        ax[i,j].set_yticks(x)
        ax[i,j].set_yticklabels([])
        # Add color patches in place of x-tick labels
        for k, color in enumerate(colors):
            rect = patches.Rectangle((x[k]-0.03,-0.08), 0.06, 0.06, facecolor=color, transform=ax[i,j].transData, clip_on=False)
            ax[i,j].add_patch(rect)
            rect = patches.Rectangle((-0.08,x[k]-0.03), 0.06, 0.06, facecolor=color, transform=ax[i,j].transData, clip_on=False)
            ax[i,j].add_patch(rect)
        # plt.colorbar(p)
# plt.gca().invert_yaxis()
# ax[0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
# plt.xlabel("Hue 1 [deg]")
# plt.ylabel("Hue 2 [deg]")
plt.tight_layout()
# plt.savefig("hue_analysis_dkr.pdf")
# plt.show()

plt.savefig("6d_dkr_lab_hsl.pdf")
print('stop')


