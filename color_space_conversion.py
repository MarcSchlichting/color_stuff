import numpy as np
import matplotlib.pyplot as plt
from colormath.color_objects import sRGBColor, LabColor, HSLColor
from colormath.color_conversions import convert_color
import matplotlib.patches as patches
from delta_e_2000 import delta_e_2000_gpt
from colorspacious import cspace_convert

# fig,axs = plt.subplots(1,2)
# c = [(1,0,0),(0.5,0.5,0),(0,1,0),(0,0.5,0.5),(0,0,1)]
# rs = np.array([10,20,30,50,80])
rs = np.linspace(5,90,20)
a_ts = []
b_ts = []
for i in range(rs.shape[0]):

    L = 20
    r = rs[i]
    t = np.linspace(0,2*np.pi,100)
    a = r * np.cos(t)
    b = r * np.sin(t)

    reference_color = np.array([[L,0,0]])

    lab = np.stack([L*np.ones_like(a),a,b],axis=-1)
    reference_color = np.repeat(reference_color,lab.shape[0],axis=0)
    de = delta_e_2000_gpt(reference_color,lab)

    a_tilde = de * np.cos(t)
    b_tilde = de * np.sin(t)
    
    a_ts.append(a_tilde)
    b_ts.append(b_tilde)


    # axs[0].scatter(a,b,s=5,color=c[i])
    # axs[1].scatter(a_tilde,b_tilde,s=5,color=c[i])
    # axs[0].axis("equal")
    # axs[1].axis("equal")

# ratio = np.array([np.sqrt(a_ts[i]**2+b_ts[i]**2)/np.sqrt(a**2 + b**2) for i in range(rs.shape[0])]).mean(axis=1)
# plt.plot(rs,ratio)
# plt.show()
# range = np.array([np.sqrt(a_ts[i]**2+b_ts[i]**2)/np.sqrt(a**2 + b**2) for i in range(rs.shape[0])])
# amplitutde = range.max(axis=1) - range.min(axis=1)
# plt.plot(rs,amplitutde)
# plt.show()
for i in range(rs.shape[0]):
    plt.plot(t,np.sqrt(a_ts[i]**2+b_ts[i]**2)/np.sqrt(a**2 + b**2))
plt.show()


print("stop")