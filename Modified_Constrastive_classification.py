import torch
import pandas as pd
import ast
from colormath.color_objects import sRGBColor, HSLColor
from colormath.color_conversions import convert_color

class Encoder(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.l1 = torch.nn.Linear(6,8)
        self.l2 = torch.nn.Linear(8,8)
        self.l3 = torch.nn.Linear(8,8)
        self.l4 = torch.nn.Linear(8,8)
        self.l5 = torch.nn.Linear(8,8)
        self.l6 = torch.nn.Linear(8,8)
        self.l7 = torch.nn.Linear(8,4)
        self.l8 = torch.nn.Linear(4,2)
        self.l9 = torch.nn.Linear(2,1)

    
    def forward(self,x):
        z = self.l1(x)
        z = torch.relu(z)
        # z = self.l2(z)
        # z = torch.relu(z)
        # z = self.l3(z)
        # z = torch.relu(z)
        # z = self.l4(z)
        # z = torch.relu(z)
        # z = self.l5(z)
        # z = torch.relu(z)
        # z = self.l6(z)
        # z = torch.relu(z)
        z = self.l7(z)
        z = torch.relu(z)
        z = self.l8(z)
        z = torch.relu(z)
        z = self.l9(z)

        return torch.nn.functional.softplus(z)


df = pd.read_csv("preferences.csv")

all_ap = []
all_an = []


for i in range(df.shape[0]):
    anchor_rgb = torch.Tensor(ast.literal_eval(df["Anchor_RGB"].iloc[i]))
    A_rgb = torch.Tensor(ast.literal_eval(df["Color_A_RGB"].iloc[i]))
    B_rgb = torch.Tensor(ast.literal_eval(df["Color_B_RGB"].iloc[i]))
    preference = df["Preference"].iloc[i]

    anchor_hsl = torch.Tensor(convert_color(sRGBColor(*anchor_rgb),HSLColor).get_value_tuple())
    A_hsl = torch.Tensor(convert_color(sRGBColor(*A_rgb),HSLColor).get_value_tuple())
    B_hsl = torch.Tensor(convert_color(sRGBColor(*B_rgb),HSLColor).get_value_tuple())

    if preference == "A":
        all_ap.append(torch.concatenate([anchor_hsl,A_hsl]))
        all_ap.append(torch.concatenate([A_hsl,anchor_hsl]))
        all_an.append(torch.concatenate([anchor_hsl,B_hsl]))
        all_an.append(torch.concatenate([B_hsl,anchor_hsl]))


    elif preference == "B":
        all_ap.append(torch.concatenate([anchor_hsl,B_hsl]))
        all_ap.append(torch.concatenate([B_hsl,anchor_hsl]))
        all_an.append(torch.concatenate([anchor_hsl,A_hsl]))
        all_an.append(torch.concatenate([A_hsl,anchor_hsl]))
    

all_ap = torch.stack(all_ap,dim=0) *torch.Tensor([1/180,2,2,1/180,2,2]) - 1        # normalize to -1, 1
all_an = torch.stack(all_an,dim=0) *torch.Tensor([1/180,2,2,1/180,2,2]) - 1      # normalize to -1, 1

#split dataset
all_idx = torch.randperm(all_ap.shape[0])
train_fraction = 0.7
train_idx = all_idx[:int(train_fraction*all_ap.shape[0])]
test_idx = all_idx[int(train_fraction*all_ap.shape[0]):]

train_ap = all_ap[train_idx]
train_an = all_an[train_idx]

test_ap = all_ap[test_idx]
test_an = all_an[test_idx]

model = Encoder()
optim = torch.optim.Adam(model.parameters(),lr=0.0001)

def modified_triplet_loss(d_ap,d_an):
    return torch.mean(torch.max(torch.zeros_like(d_ap),d_ap-d_an+2.5)**2)

criterion = modified_triplet_loss

n_epochs = 100000

for e in range(n_epochs):
    optim.zero_grad()
    d_ap_train = model(train_ap)
    d_an_train = model(train_an)

    train_loss = criterion(d_ap_train,d_an_train)
    

    train_loss.backward()
    optim.step()

    with torch.no_grad():
        d_ap_test = model(test_ap)
        d_an_test = model(test_an)

        test_loss = criterion(d_ap_test,d_an_test)

        train_fraction = (d_ap_train < d_an_train).to(torch.float64).mean().item()
        test_fraction = (d_ap_test < d_an_test).to(torch.float64).mean().item()
 
    
    if (e%100)==0:
        # print(f"{e}/{n_epochs} - Train Loss: {train_loss.detach().item()} - APR Train: {ap_fraction_train} - ANR Train: {an_fraction_train} - Test Loss: {test_loss.detach().item()} - APR Test: {ap_fraction_test} - ANR Test: {an_fraction_test}")
        print(f"{e}/{n_epochs} - Train Acc: {train_fraction} - Test_Acc: {test_fraction}")
    
    if (train_fraction >= 0.85) and (test_fraction >= 0.85):
        break


print("stop")
