import numpy as np

def sind(x):
    return np.sin(np.deg2rad(x))

def cosd(x):
    return np.cos(np.deg2rad(x))

def atan2d(y,x):
    return np.rad2deg(np.arctan2(y,x))

import math

def delta_e_2000_gpt(lab1, lab2):
    def deg_to_rad(deg):
        return deg * (np.pi / 180.0)

    def rad_to_deg(rad):
        return rad * (180.0 / np.pi)

    def CIEDE2000(L1, a1, b1, L2, a2, b2):
        L1, a1, b1 = np.asarray(L1), np.asarray(a1), np.asarray(b1)
        L2, a2, b2 = np.asarray(L2), np.asarray(a2), np.asarray(b2)

        L_bar_prime = 0.5 * (L1 + L2)
        
        C1 = np.sqrt(a1**2 + b1**2)
        C2 = np.sqrt(a2**2 + b2**2)
        
        C_bar = 0.5 * (C1 + C2)
        
        C_bar_prime = 1.0 + 0.5 * (1.0 - np.sqrt(C_bar**7 / (C_bar**7 + 25.0**7)))
        
        a1_prime = a1 * C_bar_prime
        a2_prime = a2 * C_bar_prime
        
        C1_prime = np.sqrt(a1_prime**2 + b1**2)
        C2_prime = np.sqrt(a2_prime**2 + b2**2)
        
        C_bar_prime = 0.5 * (C1_prime + C2_prime)
        
        h1_prime = np.arctan2(b1, a1_prime)
        h2_prime = np.arctan2(b2, a2_prime)
        h1_prime = np.where(h1_prime >= 0, h1_prime, h1_prime + 2 * np.pi)
        h2_prime = np.where(h2_prime >= 0, h2_prime, h2_prime + 2 * np.pi)
        
        H_bar_prime = 0.5 * (h1_prime + h2_prime)
        H_bar_prime = np.where(np.abs(h1_prime - h2_prime) > np.pi, 
                               H_bar_prime + np.pi, H_bar_prime)
        H_bar_prime = np.where(H_bar_prime < 2 * np.pi, H_bar_prime, H_bar_prime - 2 * np.pi)
        
        T = (1 -
             0.17 * np.cos(H_bar_prime - deg_to_rad(30)) +
             0.24 * np.cos(2 * H_bar_prime) +
             0.32 * np.cos(3 * H_bar_prime + deg_to_rad(6)) -
             0.20 * np.cos(4 * H_bar_prime - deg_to_rad(63)))
        
        delta_h_prime = h2_prime - h1_prime
        delta_h_prime = np.where(np.abs(delta_h_prime) > np.pi,
                                 np.where(h2_prime <= h1_prime, 
                                          delta_h_prime + 2 * np.pi, 
                                          delta_h_prime - 2 * np.pi),
                                 delta_h_prime)
        
        delta_L_prime = L2 - L1
        delta_C_prime = C2_prime - C1_prime
        delta_H_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(delta_h_prime / 2)
        
        S_L = 1 + (0.015 * (L_bar_prime - 50) ** 2) / np.sqrt(20 + (L_bar_prime - 50) ** 2)
        S_C = 1 + 0.045 * C_bar_prime
        S_H = 1 + 0.015 * C_bar_prime * T
        
        delta_theta = deg_to_rad(30) * np.exp(-((H_bar_prime - deg_to_rad(275)) / deg_to_rad(25)) ** 2)
        
        R_C = 2 * np.sqrt(C_bar_prime ** 7 / (C_bar_prime ** 7 + 25.0 ** 7))
        R_T = -R_C * np.sin(2 * delta_theta)
        
        delta_E_00 = np.sqrt((delta_L_prime / S_L) ** 2 +
                             (delta_C_prime / S_C) ** 2 +
                             (delta_H_prime / S_H) ** 2 +
                             R_T * (delta_C_prime / S_C) * (delta_H_prime / S_H))
        
        return delta_E_00

    L1, a1, b1 = lab1.T
    L2, a2, b2 = lab2.T
    
    return CIEDE2000(L1, a1, b1, L2, a2, b2)


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
    
    print(delta_e_cie2000(LabColor(*lab1),LabColor(*lab2)))
    print(delta_e_cie2000(LabColor(*lab2),LabColor(*lab3)))
    print(delta_e_cie2000(LabColor(*lab3),LabColor(*lab4)))
    # print(delta_e_2000(lab3,lab4))
    print(delta_e_2000_gpt(np.stack([lab1,lab2,lab3],axis=0),np.stack([lab2,lab3,lab4],axis=0)))
    print('stop')