import pandas as pd
import pickle
import numpy as np


df = pd.read_csv("responses.csv",header=1)
df = df.drop(0) #delete the weird row
df = df.reset_index(drop=True)

delta_e_levels = np.arange(15,71,5)
question_idx = np.arange(200)

impossible_answers = 0
all_responses = pd.DataFrame(columns=["Anchor_RGB","Color_A_RGB","Color_B_RGB","Preference"])
counter = 0
for de in delta_e_levels:
    with open(f"./candidate_colors/{de}/colors_{de}.pkl","rb") as f:
        colors = pickle.load(f)
    for qi in question_idx:
        q_id_str = f"QID_{de}_{qi}"
        responses = df[q_id_str][~df[q_id_str].isna()]
        c = colors[qi]
        if responses.unique().shape[0]>1:
            impossible_answers += responses.shape[0]/2
        for r in range(responses.shape[0]):
            answer = responses.iloc[r]
            row = {"Anchor_RGB":[list(c["anchor"])], "Color_A_RGB":[list(c["c1"])], "Color_B_RGB":[list(c["c2"])], "Preference":answer}
            all_responses = pd.concat([all_responses,pd.DataFrame(row)],axis=0,ignore_index=True)
            counter += 1

        
all_responses.to_csv("preferences.csv",index=None)

print("Done.")