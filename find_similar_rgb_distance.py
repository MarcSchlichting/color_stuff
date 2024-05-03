from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

import numpy as np
from tqdm import tqdm

def patch_asscalar(a):
    return a.item()

setattr(np, "asscalar", patch_asscalar)

highest_delta_e = 0
best_color1 = None
best_color2 = None

lowest_delta_e = 1000
worst_color1 = None
worst_color2 = None


for _ in tqdm(range(10000)):

    color1 = np.random.rand(3)
    diff = 0.3
    random_diff = np.random.rand(3)
    color2 = color1 + diff*random_diff/np.linalg.norm(random_diff)
    if np.all(color2<=1) and np.all(color2>0):  
        delta_e = delta_e_cie2000(convert_color(sRGBColor(*color1),LabColor),convert_color(sRGBColor(*color2),LabColor))
        if delta_e>highest_delta_e:
            highest_delta_e = delta_e
            best_color1 = color1
            best_color2 = color2
        if delta_e<lowest_delta_e:
            lowest_delta_e = delta_e
            worst_color1 = color1
            worst_color2 = color2

print(best_color1*255,best_color2*255,highest_delta_e)
print(best_color1,best_color2,highest_delta_e)
print(worst_color1*255,worst_color2*255,lowest_delta_e)
print(worst_color1,worst_color2,lowest_delta_e)
