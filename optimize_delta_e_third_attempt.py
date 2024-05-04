import numpy as np
from scipy.optimize import minimize

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colorspacious import cspace_convert
from tqdm import tqdm
import matplotlib.pyplot as plt

def patch_asscalar(a):
    return a.item()

setattr(np, "asscalar", patch_asscalar)

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

def convert_to_rgb(lab):
    rgb = convert_color(LabColor(*lab),sRGBColor)
    return [rgb.rgb_r,rgb.rgb_g,rgb.rgb_b]

# Define the objective function
def objective_function(x):

    # Normal vision
    color1 = x[:3]
    color2 = x[3:6]
    color3 = x[6:]
    background= [1,1,1]

    delta_e_12 = delta_e_cie2000(convert_color(sRGBColor(*color1),LabColor),convert_color(sRGBColor(*color2),LabColor))
    delta_e_23 = delta_e_cie2000(convert_color(sRGBColor(*color2),LabColor),convert_color(sRGBColor(*color3),LabColor))
    delta_e_13 = delta_e_cie2000(convert_color(sRGBColor(*color1),LabColor),convert_color(sRGBColor(*color3),LabColor))

    delta_e_01 = delta_e_cie2000(convert_color(sRGBColor(*color1),LabColor),convert_color(sRGBColor(*background),LabColor))
    delta_e_02 = delta_e_cie2000(convert_color(sRGBColor(*color2),LabColor),convert_color(sRGBColor(*background),LabColor))
    delta_e_03 = delta_e_cie2000(convert_color(sRGBColor(*color3),LabColor),convert_color(sRGBColor(*background),LabColor))
    
    # sum_normal = delta_e_12 + delta_e_13 + delta_e_23 + 1*(delta_e_01 + delta_e_02 + delta_e_03)
    sum_normal = np.min([delta_e_12,delta_e_13,delta_e_23,delta_e_01,delta_e_02,delta_e_03])

    # Protanopia
    color1_p = simulate_colorblindness(color1,"p")
    color2_p = simulate_colorblindness(color2,"p")
    color3_p = simulate_colorblindness(color3,"p")
    background_p = simulate_colorblindness(background,"p")

    delta_e_12_p = delta_e_cie2000(convert_color(sRGBColor(*color1_p),LabColor),convert_color(sRGBColor(*color2_p),LabColor))
    delta_e_23_p = delta_e_cie2000(convert_color(sRGBColor(*color2_p),LabColor),convert_color(sRGBColor(*color3_p),LabColor))
    delta_e_13_p = delta_e_cie2000(convert_color(sRGBColor(*color1_p),LabColor),convert_color(sRGBColor(*color3_p),LabColor))

    delta_e_01_p = delta_e_cie2000(convert_color(sRGBColor(*color1_p),LabColor),convert_color(sRGBColor(*background_p),LabColor))
    delta_e_02_p = delta_e_cie2000(convert_color(sRGBColor(*color2_p),LabColor),convert_color(sRGBColor(*background_p),LabColor))
    delta_e_03_p = delta_e_cie2000(convert_color(sRGBColor(*color3_p),LabColor),convert_color(sRGBColor(*background_p),LabColor))

    # sum_p = delta_e_12_p + delta_e_13_p + delta_e_23_p + 1*(delta_e_01_p + delta_e_02_p + delta_e_03_p)
    sum_p = np.min([delta_e_12_p,delta_e_13_p,delta_e_23_p,delta_e_01_p,delta_e_02_p,delta_e_03_p])


    # Deut.
    color1_d = simulate_colorblindness(color1,"d")
    color2_d = simulate_colorblindness(color2,"d")
    color3_d = simulate_colorblindness(color3,"d")
    background_d = simulate_colorblindness(background,"d")

    delta_e_12_d = delta_e_cie2000(convert_color(sRGBColor(*color1_d),LabColor),convert_color(sRGBColor(*color2_d),LabColor))
    delta_e_23_d = delta_e_cie2000(convert_color(sRGBColor(*color2_d),LabColor),convert_color(sRGBColor(*color3_d),LabColor))
    delta_e_13_d = delta_e_cie2000(convert_color(sRGBColor(*color1_d),LabColor),convert_color(sRGBColor(*color3_d),LabColor))

    delta_e_01_d = delta_e_cie2000(convert_color(sRGBColor(*color1_d),LabColor),convert_color(sRGBColor(*background_d),LabColor))
    delta_e_02_d = delta_e_cie2000(convert_color(sRGBColor(*color2_d),LabColor),convert_color(sRGBColor(*background_d),LabColor))
    delta_e_03_d = delta_e_cie2000(convert_color(sRGBColor(*color3_d),LabColor),convert_color(sRGBColor(*background_d),LabColor))

    # sum_d = delta_e_12_d + delta_e_13_d + delta_e_23_d + 1*(delta_e_01_d + delta_e_02_d + delta_e_03_d)
    sum_d = np.min([delta_e_12_d,delta_e_13_d,delta_e_23_d,delta_e_01_d,delta_e_02_d,delta_e_03_d])


    # Tri.
    color1_t = simulate_colorblindness(color1,"t")
    color2_t = simulate_colorblindness(color2,"t")
    color3_t = simulate_colorblindness(color3,"t")
    background_t = simulate_colorblindness(background,"t")

    delta_e_12_t = delta_e_cie2000(convert_color(sRGBColor(*color1_t),LabColor),convert_color(sRGBColor(*color2_t),LabColor))
    delta_e_23_t = delta_e_cie2000(convert_color(sRGBColor(*color2_t),LabColor),convert_color(sRGBColor(*color3_t),LabColor))
    delta_e_13_t = delta_e_cie2000(convert_color(sRGBColor(*color1_t),LabColor),convert_color(sRGBColor(*color3_t),LabColor))

    delta_e_01_t = delta_e_cie2000(convert_color(sRGBColor(*color1_t),LabColor),convert_color(sRGBColor(*background_t),LabColor))
    delta_e_02_t = delta_e_cie2000(convert_color(sRGBColor(*color2_t),LabColor),convert_color(sRGBColor(*background_t),LabColor))
    delta_e_03_t = delta_e_cie2000(convert_color(sRGBColor(*color3_t),LabColor),convert_color(sRGBColor(*background_t),LabColor))

    # sum_t = delta_e_12_t + delta_e_13_t + delta_e_23_t + 1*(delta_e_01_t + delta_e_02_t + delta_e_03_t)
    sum_t = np.min([delta_e_12_t,delta_e_13_t,delta_e_23_t,delta_e_01_t,delta_e_02_t,delta_e_03_t])

    # return - np.min([sum_normal,sum_p,sum_d,sum_t])
    return -sum_normal - sum_p - sum_d - sum_t
    # return -sum_normal - sum_d 

# Define the initial guess (starting point)
initial_guess = np.random.uniform(0, 1, size=9)

# Define any constraints (if applicable)
# Example: bounds on variables
bounds = [(0,1) for _ in range(9)]

# Perform optimization
result = minimize(objective_function, initial_guess, bounds=bounds)

# #MC variant
# xs = []
# scores = []
# for _ in tqdm(range(10000)):
#     x = np.random.uniform(0, 1, size=9)
#     score = -objective_function(x)
#     xs.append(x)
#     scores.append(score)

# x = xs[np.argmax(scores)]

# plt.scatter([0],[0],s=500,c=x[:3])
# plt.scatter([1],[1],s=500,c=x[3:6])
# plt.scatter([2],[2],s=500,c=x[6:])
# plt.show()
# # print(resusuccess)
# print(np.array(x[:3])*255)
# print(np.array(x[3:6])*255)
# print(np.array(x[6:])*255)

plt.scatter([0],[0],s=500,c=result.x[:3])
plt.scatter([1],[1],s=500,c=result.x[3:6])
plt.scatter([2],[2],s=500,c=result.x[6:])

plt.scatter([1],[0],s=500,c=simulate_colorblindness(result.x[:3],"p"))
plt.scatter([2],[1],s=500,c=simulate_colorblindness(result.x[3:6],"p"))
plt.scatter([3],[2],s=500,c=simulate_colorblindness(result.x[6:],"p"))

plt.scatter([2],[0],s=500,c=simulate_colorblindness(result.x[:3],"d"))
plt.scatter([3],[1],s=500,c=simulate_colorblindness(result.x[3:6],"d"))
plt.scatter([4],[2],s=500,c=simulate_colorblindness(result.x[6:],"d"))

plt.scatter([3],[0],s=500,c=simulate_colorblindness(result.x[:3],"t"))
plt.scatter([4],[1],s=500,c=simulate_colorblindness(result.x[3:6],"t"))
plt.scatter([5],[2],s=500,c=simulate_colorblindness(result.x[6:],"t"))

plt.show()
print(result.success)
print(result.fun)
print("normal")
print(np.array(result.x[:3])*255)
print(np.array(result.x[3:6])*255)
print(np.array(result.x[6:])*255)
print("pro")
print(np.array(simulate_colorblindness(result.x[:3],type="p"))*255)
print(np.array(simulate_colorblindness(result.x[3:6],type="p"))*255)
print(np.array(simulate_colorblindness(result.x[6:],type="p"))*255)
print("deut")
print(np.array(simulate_colorblindness(result.x[:3],type="d"))*255)
print(np.array(simulate_colorblindness(result.x[3:6],type="d"))*255)
print(np.array(simulate_colorblindness(result.x[6:],type="d"))*255)
print("tri")
print(np.array(simulate_colorblindness(result.x[:3],type="t"))*255)
print(np.array(simulate_colorblindness(result.x[3:6],type="t"))*255)
print(np.array(simulate_colorblindness(result.x[6:],type="t"))*255)

print("stop")