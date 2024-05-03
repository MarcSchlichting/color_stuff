import numpy as np
from scipy.optimize import minimize

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from tqdm import tqdm

def patch_asscalar(a):
    return a.item()

setattr(np, "asscalar", patch_asscalar)

def convert_to_rgb(lab):
    rgb = convert_color(LabColor(*lab),sRGBColor)
    return [rgb.rgb_r,rgb.rgb_g,rgb.rgb_b]

# Define the objective function
def objective_function(x):
    color1 = x[:3]
    color2 = x[3:6]
    color3 = x[6:]
    background= [0,0,0]

    delta_e_12 = delta_e_cie2000(convert_color(sRGBColor(*color1),LabColor),convert_color(sRGBColor(*color2),LabColor))
    delta_e_23 = delta_e_cie2000(convert_color(sRGBColor(*color2),LabColor),convert_color(sRGBColor(*color3),LabColor))
    delta_e_13 = delta_e_cie2000(convert_color(sRGBColor(*color1),LabColor),convert_color(sRGBColor(*color3),LabColor))

    delta_e_01 = delta_e_cie2000(convert_color(sRGBColor(*color1),LabColor),convert_color(sRGBColor(*background),LabColor))
    delta_e_02 = delta_e_cie2000(convert_color(sRGBColor(*color2),LabColor),convert_color(sRGBColor(*background),LabColor))
    delta_e_03 = delta_e_cie2000(convert_color(sRGBColor(*color3),LabColor),convert_color(sRGBColor(*background),LabColor))

    return -delta_e_12 - delta_e_13 - delta_e_23 - delta_e_01 - delta_e_02 - delta_e_03

# Define the initial guess (starting point)
initial_guess = np.random.uniform(0.9, 0.1, size=9)

# Define any constraints (if applicable)
# Example: bounds on variables
bounds = [(0,1) for _ in range(9)]

# Perform optimization
result = minimize(objective_function, initial_guess, bounds=bounds)

print(result.success)
print(np.array(result.x[:3])*255)
print(np.array(result.x[3:6])*255)
print(np.array(result.x[6:])*255)
print("stop")