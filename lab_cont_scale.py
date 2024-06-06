import numpy as np
import matplotlib.pyplot as plt
from colormath.color_objects import sRGBColor, LabColor, HSLColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import matplotlib.patches as patches

def patch_asscalar(a):
    return a.item()

setattr(np, "asscalar", patch_asscalar)

def plot_color_sequence(cs):
    fig,ax = plt.subplots(1,1)

    for i,c in enumerate(cs):
        rect = patches.Rectangle((i,0), 0.8, 0.8, facecolor=c, transform=ax.transData, clip_on=False)
        ax.add_patch(rect)
    plt.xlim(-0.5,len(cs))
    ax.axis("equal")
    ax.axis("off")
    plt.show()
    print("stop")

lab1 = np.array([50,-110,45])
rgb1 = np.clip(convert_color(LabColor(*lab1),sRGBColor).get_value_tuple(),0,1)
lab2 = np.array([50,105,-98])
rgb2 = np.clip(convert_color(LabColor(*lab2),sRGBColor).get_value_tuple(),0,1)




xs = np.linspace(0,1,11)

labs = [x * lab1 + (1-x)*lab2 for x in xs]
rgbs = [np.clip(convert_color(LabColor(*lab),sRGBColor).get_value_tuple(),0,1) for lab in labs]
rgbs2 = [x * rgb1 + (1-x)*rgb2 for x in xs]



plot_color_sequence(rgbs)