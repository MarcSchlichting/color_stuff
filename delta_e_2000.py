import numpy as np

def sind(x):
    return np.sin(np.deg2rad(x))

def cosd(x):
    return np.cos(np.deg2rad(x))

def atan2d(y,x):
    return np.rad2deg(np.arctan2(y,x))

# def delta_e_2000(lab1,lab2):
#     """expects lab1 and lab2 to be of shape (3,N)"""
#     assert lab1.shape == lab2.shape
#     assert lab1.shape[0] == 3
    
#     k_L = 1     # ISO 13655:2009
#     delta_Lp = lab2[0] - lab1[0]
#     Lbar = 0.5 * (lab1[0]+lab2[0])
#     S_L = 1 + (0.015*(Lbar-50)**2)/np.sqrt(20+(Lbar-50)**2)
    
#     C_1s = np.sqrt(lab1[1]**2 + lab1[2]**2)
#     C_2s = np.sqrt(lab2[1]**2 + lab2[2]**2)
#     Cbar = (C_1s + C_2s)/2
#     a_1p = lab1[1] + lab1[1]/2*(1-np.sqrt(Cbar**7/(Cbar**7 + 25**7)))
#     a_2p = lab2[1] + lab2[1]/2*(1-np.sqrt(Cbar**7/(Cbar**7 + 25**7)))
#     C_1p = np.sqrt(a_1p**2 + lab1[2]**2)
#     C_2p = np.sqrt(a_2p**2 + lab2[2]**2)
#     delta_Cp = C_2p - C_1p
#     k_C = 1     # ISO 13655:2009
    
#     Cbarp = (C_1p + C_2p)/2
#     S_C = 1 + 0.045*Cbarp
    
#     h_1p = np.mod(atan2d(lab1[2],a_1p),360)
#     h_2p = np.mod(atan2d(lab2[2],a_2p),360)
#     # delta_hp = np.mod((h_2p - h_1p)+180, 360) - 180
#     delta_hp = np.where(np.abs(h_1p - h_2p) <= 180, h_2p - h_1p, (360 - np.abs(h_2p - h_1p))) 

#     delta_Hp = 2*np.sqrt(C_1p*C_2p)*sind(delta_hp/2)
#     k_H = 1     # ISO 13655:2009
    
#     Hbarp = np.where(np.abs(h_1p - h_2p) > 180, (h_1p + h_2p + 360)/2, (h_1p + h_2p)/2)
#     T = 1 - 0.17*cosd(Hbarp-30) + 0.24*cosd(2*Hbarp) + 0.32*cosd(2*Hbarp + 6) -0.2*cosd(4*Hbarp - 63)
#     S_H = 1 + 0.015*Cbarp*T    
    
#     R_T = -2*np.sqrt(Cbarp**7/(Cbarp**7 + 25**7))*sind(60*np.exp(-((Hbarp-275)/25)**2))
    
#     delta_e = np.sqrt((delta_Lp/(k_L*S_L))**2+(delta_Cp/(k_C*S_C))**2+(delta_Hp/(k_H*S_H))**2+R_T*delta_Cp/(k_C*S_C)*delta_Hp/(k_H*S_H))
    
#     return delta_e

def delta_e_2000(lab1, lab2):
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    
    # Calculate C1, C2
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    
    # Calculate C_ab_bar
    C_ab_bar = (C1 + C2) / 2
    
    # Calculate G
    G = 0.5 * (1 - np.sqrt(C_ab_bar**7 / (C_ab_bar**7 + 25**7)))
    
    # Calculate a1', a2'
    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2
    
    # Calculate C1', C2'
    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)
    
    # Calculate h1', h2'
    h1_prime = np.degrees(np.arctan2(b1, a1_prime))
    h2_prime = np.degrees(np.arctan2(b2, a2_prime))
    
    # Ensure h1_prime, h2_prime are within [0, 360)
    h1_prime = np.where(h1_prime >= 0, h1_prime, h1_prime + 360)
    h2_prime = np.where(h2_prime >= 0, h2_prime, h2_prime + 360)
    
    # Calculate delta L', delta C', delta h', delta H'
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    
    h_diff = h2_prime - h1_prime
    h_sum = h1_prime + h2_prime
    
    h_diff = np.where(np.abs(h_diff) <= 180, h_diff, np.where(h_diff > 0, h_diff - 360, h_diff + 360))
    h_sum = np.where(h_diff != 0, (h1_prime + h2_prime) / 2, h1_prime)
    
    delta_h_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(h_diff) / 2)
    
    # Calculate C_ab_bar_prime
    C_ab_bar_prime = (C1_prime + C2_prime) / 2
    
    # Calculate h_bar_prime
    h_bar_prime = np.where(np.abs(h_diff) <= 180, h_sum, h_sum + 180)
    
    # Calculate T
    T = 1 - 0.17 * np.cos(np.radians(h_bar_prime - 30)) + 0.24 * np.cos(np.radians(2 * h_bar_prime)) + 0.32 * np.cos(np.radians(3 * h_bar_prime + 6)) - 0.20 * np.cos(np.radians(4 * h_bar_prime - 63))
    
    # Calculate delta theta
    delta_theta = 30 * np.exp(-((h_bar_prime - 275) / 25)**2)
    
    # Calculate R_C
    R_C = 2 * np.sqrt(C_ab_bar_prime**7 / (C_ab_bar_prime**7 + 25**7)) * np.sin(np.radians(delta_theta))
    
    # Calculate S_L, S_C, S_H
    S_L = 1 + (0.015 * (L1 - 50)**2) / np.sqrt(20 + (L1 - 50)**2)
    S_C = 1 + 0.045 * C_ab_bar_prime
    S_H = 1 + 0.015 * C_ab_bar_prime * T
    
    # Calculate delta E 2000
    delta_E_2000 = np.sqrt((delta_L_prime / S_L)**2 + (delta_C_prime / S_C)**2 + (delta_h_prime / S_H)**2 + R_C * (delta_C_prime / S_C) * (delta_h_prime / S_H))
    
    return delta_E_2000

if __name__ == "__main__":
    from colormath.color_diff import delta_e_cie2000
    from colormath.color_objects import sRGBColor, LabColor
    
    def patch_asscalar(a):
        return a.item()

    setattr(np, "asscalar", patch_asscalar)
    
    lab1 = np.array([50,-13,56])
    lab2 = np.array([13,90,66])
    lab3 = np.array([77,-33,69])
    lab4 = np.array([34,4,-111])
    
    print(delta_e_cie2000(LabColor(*lab3),LabColor(*lab4)))
    print(delta_e_2000(lab3,lab4))