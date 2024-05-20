import torch
import pandas as pd
import ast
from colormath.color_objects import sRGBColor, HSLColor
from colormath.color_conversions import convert_color

class Encoder(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.l1 = torch.nn.Linear(3,8)
        self.l2 = torch.nn.Linear(8,8)
        self.l3 = torch.nn.Linear(8,8)
        self.l4 = torch.nn.Linear(8,8)
        self.l5 = torch.nn.Linear(8,8)
        self.l6 = torch.nn.Linear(8,8)
        self.l7 = torch.nn.Linear(8,8)
        self.l8 = torch.nn.Linear(8,8)
        self.l9 = torch.nn.Linear(8,3)

    
    def forward(self,x):
        z = self.l1(x)
        z = torch.relu(z)
        z = self.l2(z)
        z = torch.relu(z)
        z = self.l3(z)
        z = torch.relu(z)
        z = self.l4(z)
        z = torch.relu(z)
        z = self.l5(z)
        z = torch.relu(z)
        z = self.l6(z)
        z = torch.relu(z)
        z = self.l7(z)
        z = torch.relu(z)
        z = self.l8(z)
        z = torch.relu(z)
        z = self.l9(z)

        return z


df = pd.read_csv("preferences.csv")

all_anchors = []
all_positives = []
all_negatives = []

for i in range(df.shape[0]):
    anchor_rgb = torch.Tensor(ast.literal_eval(df["Anchor_RGB"].iloc[i]))
    A_rgb = torch.Tensor(ast.literal_eval(df["Color_A_RGB"].iloc[i]))
    B_rgb = torch.Tensor(ast.literal_eval(df["Color_B_RGB"].iloc[i]))
    preference = df["Preference"].iloc[i]

    anchor_hsl = torch.Tensor(convert_color(sRGBColor(*anchor_rgb),HSLColor).get_value_tuple())
    A_hsl = torch.Tensor(convert_color(sRGBColor(*A_rgb),HSLColor).get_value_tuple())
    B_hsl = torch.Tensor(convert_color(sRGBColor(*B_rgb),HSLColor).get_value_tuple())

    # all_anchors.append(anchor_rgb)
    all_anchors.append(anchor_hsl)

    # if preference == "A":
    #     all_positives.append(A_rgb)
    #     all_negatives.append(B_rgb)

    # elif preference == "B":
    #     all_positives.append(B_rgb)
    #     all_negatives.append(A_rgb)

    if preference == "A":
        all_positives.append(A_hsl)
        all_negatives.append(B_hsl)

    elif preference == "B":
        all_positives.append(B_hsl)
        all_negatives.append(A_hsl)
    

# all_anchors = torch.stack(all_anchors,dim=0) * 2 - 1        # normalize to -1, 1
# all_positives = torch.stack(all_positives,dim=0) * 2 -1     # normalize to -1, 1
# all_negatives = torch.stack(all_negatives,dim=0) * 2 - 1    # normalize to -1, 1

all_anchors = torch.stack(all_anchors,dim=0) *torch.Tensor([1/180,2,2]) - 1        # normalize to -1, 1
all_positives = torch.stack(all_positives,dim=0) *torch.Tensor([1/180,2,2]) - 1      # normalize to -1, 1
all_negatives = torch.stack(all_negatives,dim=0) *torch.Tensor([1/180,2,2]) - 1     # normalize to -1, 1

#split dataset
all_idx = torch.randperm(all_anchors.shape[0])
train_fraction = 0.7
train_idx = all_idx[:int(train_fraction*all_anchors.shape[0])]
test_idx = all_idx[int(train_fraction*all_anchors.shape[0]):]

train_anchors = all_anchors[train_idx]
train_positives = all_positives[train_idx]
train_negatives = all_negatives[train_idx]

test_anchors = all_anchors[test_idx]
test_positives = all_positives[test_idx]
test_negatives = all_negatives[test_idx]

model = Encoder()
optim = torch.optim.Adam(model.parameters(),lr=0.0001)
margin = 1
p = 2
criterion = torch.nn.TripletMarginLoss(margin=margin,p=p)

n_epochs = 1000000

for e in range(n_epochs):
    optim.zero_grad()
    a_enc = model(train_anchors)
    p_enc = model(train_positives)
    n_enc = model(train_negatives)
    train_loss = criterion(a_enc,p_enc,n_enc)
    

    train_loss.backward()
    optim.step()

    with torch.no_grad():
        a_enc_test = model(test_anchors)
        p_enc_test = model(test_positives)
        n_enc_test = model(test_negatives)

        test_loss = criterion(a_enc_test,p_enc_test,n_enc_test)

        ap_distance_train = torch.linalg.norm(a_enc-p_enc,dim=1)
        an_distance_train = torch.linalg.norm(a_enc-n_enc,dim=1)
        ap_fraction_train = (ap_distance_train <= margin).to(torch.float64).mean().item()
        an_fraction_train = (an_distance_train > margin).to(torch.float64).mean().item()

        ap_distance_test = torch.linalg.norm(a_enc_test-p_enc_test,dim=1)
        an_distance_test = torch.linalg.norm(a_enc_test-n_enc_test,dim=1)
        ap_fraction_test = (ap_distance_test <= margin).to(torch.float64).mean().item()
        an_fraction_test = (an_distance_test > margin).to(torch.float64).mean().item()

    
    if (e%100)==0:
        # print(f"{e}/{n_epochs} - Train Loss: {train_loss.detach().item()} - APR Train: {ap_fraction_train} - ANR Train: {an_fraction_train} - Test Loss: {test_loss.detach().item()} - APR Test: {ap_fraction_test} - ANR Test: {an_fraction_test}")
        print(f"{e}/{n_epochs} - APR+ANR Train: {ap_fraction_train+an_fraction_train} - APR+ANR Test: {ap_fraction_test+an_fraction_test}")


print("stop")
