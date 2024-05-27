import pickle

with open(f"candidate_colors_hsl/choices3.pkl","rb") as f:
    choices = pickle.load(f)

for i,c in enumerate(choices):
    if c not in ["A","B"]:
        new_choice = input(f"{i}: Enter new choice (old choice: {c}): ")
        choices[i] = new_choice

with open(f"candidate_colors_hsl/choices3_corrected.pkl","wb") as f:
    pickle.dump(choices,f)
