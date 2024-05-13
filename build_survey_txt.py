import os
import glob


for de in reversed(range(15,71,5)):
    string_lines = ["[[AdvancedFormat]]", ""]
    files = glob.glob(f'./candidate_colors/{de}/*.png', recursive=True)
    string_lines.append(f"[[Block:Delta E {de}]]")
    string_lines.append("")
    
    for f in files:
        delta_e = f.split("/")[-2]
        sample_id = f.split("/")[-1].split(".")[0]
        
        string_lines.append("[[Question:MC:SingleAnswer:Horizontal]]")
        string_lines.append(f'QID_{delta_e}_{sample_id} <img src="https://raw.githubusercontent.com/MarcSchlichting/color_stuff/main/candidate_colors/{delta_e}/{sample_id}.png" alt="Description of the image">')
        string_lines.append("[[Choices]]")
        string_lines.append("A")
        string_lines.append("B")
        string_lines.append("")
        string_lines.append("")

    s = "\n".join(string_lines)

    with open(f"questionnaire_{de}.txt","w") as f:
        f.write(s)

print("stop")
    
    