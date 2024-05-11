import numpy as np

def sind(x):
    return np.sin(np.deg2rad(x))

def cosd(x):
    return np.cos(np.deg2rad(x))

def atan2d(y,x):
    return np.rad2deg(np.arctan2(y,x))

def delta_e_2000(lab1,lab2):
    """expects lab1 and lab2 to be of shape (3,N)"""
    assert lab1.shape == lab2.shape
    assert lab1.shape[0] == 3
    
    delta_Lp = lab2[0] - lab1[0]
    k_L = 1     # ISO 13655:2009
    delta_Lp = lab2[0] - lab1[0]
    Lbar = 0.5 * (lab1[0]+lab2[0])
    S_L = 1 + (0.015*(Lbar-50)**2)/np.sqrt(20+(Lbar-50)**2)
    
    C_1s = np.sqrt(lab1[1]**2 + lab1[2]**2)
    C_2s = np.sqrt(lab2[1]**2 + lab2[2]**2)
    Cbar = (C_1s + C_2s)/2
    a_1p = lab1[1] + lab1[1]/2*(1-np.sqrt(Cbar**7/(Cbar**7 + 25**7)))
    a_2p = lab2[1] + lab2[1]/2*(1-np.sqrt(Cbar**7/(Cbar**7 + 25**7)))
    C_1p = np.sqrt(a_1p**2 + lab1[2]**2)
    C_2p = np.sqrt(a_2p**2 + lab2[2]**2)
    delta_Cp = C_2p - C_1p
    k_C = 1     # ISO 13655:2009
    
    Cbarp = (C_1p + C_2p)/2
    S_C = 1 + 0.045*Cbarp
    
    h_1p = np.mod(atan2d(lab1[2],a_1p),360)
    h_2p = np.mod(atan2d(lab2[2],a_2p),360)
    delta_hp = np.mod((h_2p - h_1p)+180, 360) - 180
    delta_Hp = 2*np.sqrt(C_1p*C_2p)*sind(delta_hp/2)
    k_H = 1     # ISO 13655:2009
    
    Hbarp = np.where(np.abs(h_1p - h_2p) > 180, (h_1p + h_2p + 360)/2, (h_1p + h_2p)/2)
    T = 1 - 0.17*cosd(Hbarp-30) + 0.24*cosd(2*Hbarp) + 0.32*cosd(2*Hbarp + 6) -0.2*cosd(4*Hbarp - 63)
    S_H = 1 + 0.015*Cbarp*T    
    
    R_T = -2*np.sqrt(Cbar**7/(Cbar**7 + 25**7))*sind(60*np.exp(-((Hbarp-275)/25)**2))
    
    delta_e = np.sqrt((delta_Lp/(k_L*S_L))**2+(delta_Cp/(k_C*S_C))**2+(delta_Hp/(k_H*S_H))**2+R_T*delta_Cp/(k_C*S_C)*delta_Hp/(k_H*S_H))
    
    return delta_e

if __name__ == "__main__":
    lab1 = np.array([50,-13,56])
    lab2 = np.array([13,90,66])
    print(delta_e_2000(lab1,lab2))