import numpy as np
from colormath.color_objects import sRGBColor, LabColor, HSLColor
from colormath.color_conversions import convert_color
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import itertools
from delta_e_2000 import delta_e_2000_gpt
from colorspacious import cspace_convert
from tqdm import tqdm

def plot_color_sequence(cs,ax,y=0):

    for i,c in enumerate(cs):
        rect = patches.Rectangle((i,y), 0.8, 0.8, facecolor=c, transform=ax.transData, clip_on=False)
        ax.add_patch(rect)
    plt.xlim(-0.5,len(cs))
    ax.axis("equal")
    ax.axis("off")
    # plt.show()
    # print("stop")

def simulate_colorblindness(rgb,type):
    if type == "d":
        cvd_space = {"name": "sRGB1+CVD",
                "cvd_type": "deuteranomaly",
                "severity": 100}
    elif type == "p":
        cvd_space = {"name": "sRGB1+CVD",
                "cvd_type": "protanomaly",
                "severity": 100}
    elif type == "t":
        cvd_space = {"name": "sRGB1+CVD",
                "cvd_type": "tritanomaly",
                "severity": 100}
    else:
        raise NotImplementedError
    
    rgb_colorblind = cspace_convert(rgb, cvd_space, "sRGB1")
    rgb_colorblind = np.clip(rgb_colorblind, 0, 1)
    
    return rgb_colorblind
    
def get_minimal_delta_e(given_colors_indices,delta_e_matrix,delta_e_matrix_p,delta_e_matrix_d,delta_e_matrix_t):
    all_delta_e_matrix = np.stack([delta_e_matrix,delta_e_matrix_p,delta_e_matrix_d,delta_e_matrix_t],axis=-1)
    if len(given_colors_indices) == 0:
        worst_case_delta_e = np.min(all_delta_e_matrix,axis=-1)
        max_index_flat = np.argmax(worst_case_delta_e)
        max_coordinates = np.unravel_index(max_index_flat, worst_case_delta_e.shape)
        given_colors_indices.append(max_coordinates[0])
        given_colors_indices.append(max_coordinates[1])
    
    else:
        sub_matrix = all_delta_e_matrix[given_colors_indices]
        worst_delta_e = np.min(sub_matrix,axis=(0,-1))
        new_color_idx = np.argmax(worst_delta_e)    #choose the color that has the largest smallest delta e
        given_colors_indices.append(new_color_idx)
            
        
    return given_colors_indices

# edges = []
# n = 100

# # edge 1
# edges.append(np.stack([np.linspace(0,1,n),np.zeros(n),np.zeros(n)],axis=1))

# # edge 2
# edges.append(np.stack([np.ones(n),np.linspace(0,1,n),np.zeros(n)],axis=1))

# # edge 3
# edges.append(np.stack([np.linspace(0,1,n),np.ones(n),np.zeros(n)],axis=1))

# # edge 4
# edges.append(np.stack([np.zeros(n),np.linspace(0,1,n),np.zeros(n)],axis=1))

# # edge 5
# edges.append(np.stack([np.linspace(0,1,n),np.zeros(n),np.ones(n)],axis=1))

# # edge 6
# edges.append(np.stack([np.ones(n),np.linspace(0,1,n),np.ones(n)],axis=1))

# # edge 7
# edges.append(np.stack([np.linspace(0,1,n),np.ones(n),np.ones(n)],axis=1))

# # edge 8
# edges.append(np.stack([np.zeros(n),np.linspace(0,1,n),np.ones(n)],axis=1))

# # edge 9
# edges.append(np.stack([np.zeros(n),np.zeros(n),np.linspace(0,1,n)],axis=1))

# # edge 10
# edges.append(np.stack([np.ones(n),np.zeros(n),np.linspace(0,1,n)],axis=1))

# # edge 11
# edges.append(np.stack([np.ones(n),np.ones(n),np.linspace(0,1,n)],axis=1))

# # edge 12
# edges.append(np.stack([np.zeros(n),np.ones(n),np.linspace(0,1,n)],axis=1))

# edges_rgb = np.concatenate(edges,axis=0)

# edges_lab = np.array([convert_color(sRGBColor(*e),LabColor).get_value_tuple() for e in edges_rgb])


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Scatter plot of the points
# ax.scatter(*edges_lab.T, c='b', marker='o')

# # Set labels
# ax.set_xlabel('L')
# ax.set_ylabel('a')
# ax.set_zlabel('b')
fig,ax = plt.subplots(1,1)

for y,r in tqdm(enumerate(np.arange(10,91,10))):

    labs = []
    spherical = []

    # r = 90
    phi = np.linspace(np.pi/2,3/2*np.pi,70)
    theta = np.linspace(0, np.pi, 70) # 0 to pi
    for p in phi:
        for t in theta:
            L = r*np.sin(t)*np.cos(p)
            a = r*np.sin(t)*np.sin(p)
            b = r*np.cos(t)
            
            labs.append((L,a,b))
            spherical.append((r,p,t))
    labs = np.array(labs)
    shifted_labs = labs+np.array([[100,0,0]])
    spherical = np.array(spherical)

    rgbs = np.array([convert_color(LabColor(shifted_labs[i,0],shifted_labs[i,1], shifted_labs[i,2]),sRGBColor).get_value_tuple() for i in range(shifted_labs.shape[0])])
    rgb_p = simulate_colorblindness(rgbs,"p")
    rgb_d = simulate_colorblindness(rgbs,"d")
    rgb_t = simulate_colorblindness(rgbs,"t")
    lab_p = np.array([convert_color(sRGBColor(*rgb),LabColor).get_value_tuple() for rgb in rgb_p])
    lab_d = np.array([convert_color(sRGBColor(*rgb),LabColor).get_value_tuple() for rgb in rgb_d])
    lab_t = np.array([convert_color(sRGBColor(*rgb),LabColor).get_value_tuple() for rgb in rgb_t])


    mask = (rgbs > 1) | (rgbs < 0)

    # Use np.any along the rows (axis=1) to find rows that meet the condition
    rows_with_condition = np.any(mask, axis=1)

    # Use np.where to get the indices of these rows
    indices = np.where(np.logical_not(rows_with_condition))[0]

    filtered_candidates = shifted_labs[indices] 
    filtered_candidates_p = lab_p[indices] 
    filtered_candidates_d = lab_d[indices]
    filtered_candidates_t = lab_t[indices]

    combinations = np.array(list(itertools.permutations(range(filtered_candidates.shape[0]),2)))
    lab1_candidates = filtered_candidates[combinations[:,0]]
    lab2_candidates = filtered_candidates[combinations[:,1]]

    delta_e = delta_e_2000_gpt(lab1_candidates,lab2_candidates)
    delta_e_matrix = np.zeros((filtered_candidates.shape[0],filtered_candidates.shape[0]))
    delta_e_matrix[combinations[:,0],combinations[:,1]] = delta_e

    #colorblindness
    lab1_candidates_p = filtered_candidates_p[combinations[:,0]]
    lab2_candidates_p = filtered_candidates_p[combinations[:,1]]
    delta_e_p = delta_e_2000_gpt(lab1_candidates_p,lab2_candidates_p)
    delta_e_matrix_p = np.zeros((filtered_candidates_p.shape[0],filtered_candidates_p.shape[0]))
    delta_e_matrix_p[combinations[:,0],combinations[:,1]] = delta_e_p

    lab1_candidates_d = filtered_candidates_d[combinations[:,0]]
    lab2_candidates_d = filtered_candidates_d[combinations[:,1]]
    delta_e_d = delta_e_2000_gpt(lab1_candidates_d,lab2_candidates_d)
    delta_e_matrix_d = np.zeros((filtered_candidates_d.shape[0],filtered_candidates_d.shape[0]))
    delta_e_matrix_d[combinations[:,0],combinations[:,1]] = delta_e_d

    lab1_candidates_t = filtered_candidates_t[combinations[:,0]]
    lab2_candidates_t = filtered_candidates_t[combinations[:,1]]
    delta_e_t = delta_e_2000_gpt(lab1_candidates_t,lab2_candidates_t)
    delta_e_matrix_t = np.zeros((filtered_candidates_t.shape[0],filtered_candidates_t.shape[0]))
    delta_e_matrix_t[combinations[:,0],combinations[:,1]] = delta_e_t


    color_palette = get_minimal_delta_e([],delta_e_matrix,delta_e_matrix_p, delta_e_matrix_d, delta_e_matrix_t)
    for _ in range(6):
        color_palette = get_minimal_delta_e(color_palette,delta_e_matrix,delta_e_matrix_p, delta_e_matrix_d, delta_e_matrix_t)

    lab_palette = [filtered_candidates[c] for c in color_palette]
    rgb_palette = [convert_color(LabColor(*lab),sRGBColor).get_value_tuple() for lab in lab_palette]
    plot_color_sequence(rgb_palette,ax,y=y)
plt.show()
            

    # # plot phi, theta
    # plt.scatter(spherical[indices,1],spherical[indices,2])
    # plt.xlabel(r"$\phi$")
    # plt.ylabel(r"$\theta$")
    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # Scatter plot of the points
    # ax.scatter(*shifted_labs[indices].T, c='b', marker='o')

    # # Set labels
    # ax.set_xlabel('L')
    # ax.set_ylabel('a')
    # ax.set_zlabel('b')

    # # Show the plot
    # plt.show()
print("stop")