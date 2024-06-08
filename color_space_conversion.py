import numpy as np
import matplotlib.pyplot as plt
from colormath.color_objects import sRGBColor, LabColor, HSLColor
from colormath.color_conversions import convert_color
import matplotlib.patches as patches
from delta_e_2000 import delta_e_2000_gpt
from colorspacious import cspace_convert

# l1 = np.arange(0,101,1)
# l2 = np.arange(0,101,1)
# L1,L2 = np.meshgrid(l1,l2)
# L_bar = (L1+L2)/2
# de = np.abs((L2-L1)/(1+(0.015*(L_bar-50)**2)/(np.sqrt(20+(L_bar-50)**2))))
# plt.contourf(L1,L2,de)
# plt.show()

# fig,axs = plt.subplots(1,2)
# c = [(1,0,0),(0.5,0.5,0),(0,1,0),(0,0.5,0.5),(0,0,1)]
# rs = np.array([10,20,30,50,80])
rs = np.linspace(0.001,99,20)
a_ts = []
b_ts = []
correction_factor = []
for i in range(rs.shape[0]):

    L = 20
    r = rs[i]
    t = np.linspace(0,2*np.pi,1000)
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

    correction_factor.append(de/r)


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

amplitude = [(np.max(cf)-np.min(cf))/2 for cf in correction_factor]
# scores = []
# for alpha in np.linspace(-0.01,-0.1,1000):
#     scores.append(((np.array(amplitude)-0.25*np.exp(alpha*rs))**2).mean())
# plt.plot(np.linspace(-0.01,-0.1,1000),scores)

# scores = []
# for alpha in np.linspace(-0.01,-0.1,1000):
#     scores.append(((np.array(correction_factor).mean(axis=1)-1.267*np.exp(alpha*rs))**2).mean())
# plt.plot(np.linspace(-0.01,-0.1,1000),scores)

plt.plot(rs,np.array(correction_factor).mean(axis=1))
plt.plot(rs,1.7925560597e+06*(rs+1.0090619908e+02)**(-3.1043848868e+00)+1.8062387513e-01)
plt.show()

plt.plot(rs,amplitude)
plt.plot(rs,1.974*np.exp(-((rs+100.2)/69.33)**2))
plt.show()

plt.plot(rs,(1.7925560597e+06*(rs+1.0090619908e+02)**(-3.1043848868e+00)+1.8062387513e-01) + 1.974*np.exp(-((rs+100.2)/69.33)**2)*1)
plt.plot(rs,(1.7925560597e+06*(rs+1.0090619908e+02)**(-3.1043848868e+00)+1.8062387513e-01) + 1.974*np.exp(-((rs+100.2)/69.33)**2)*0)
plt.plot(rs,(1.7925560597e+06*(rs+1.0090619908e+02)**(-3.1043848868e+00)+1.8062387513e-01) + 1.974*np.exp(-((rs+100.2)/69.33)**2)*(-1))
plt.show()

for i in range(rs.shape[0]):
    plt.plot(t,correction_factor[i])
plt.show()


print("stop")