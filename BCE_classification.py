import torch
import pandas as pd
import ast
from colormath.color_objects import sRGBColor, HSLColor
from colormath.color_conversions import convert_color
import torch.nn.backends

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device {device}.")

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
        # self.l9 = torch.nn.Linear(16,16)
        # self.l10 = torch.nn.Linear(16,16)
        # self.l11 = torch.nn.Linear(16,16)
        # self.l12 = torch.nn.Linear(16,16)
        # self.l13 = torch.nn.Linear(16,16)
        self.l14 = torch.nn.Linear(2,1)

    
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
        # z = torch.tanh(z)
        # z = self.l9(z)
        # z = torch.tanh(z)
        # z = self.l10(z)
        # z = torch.tanh(z)
        # z = self.l11(z)
        # z = torch.tanh(z)
        # z = self.l12(z)
        # z = torch.tanh(z)
        # z = self.l13(z)
        z = torch.relu(z)
        z = self.l14(z)

        return torch.sigmoid(z)


df = pd.read_csv("preferences.csv")

all_X = []
all_y = []

for i in range(df.shape[0]):
    anchor_rgb = torch.Tensor(ast.literal_eval(df["Anchor_RGB"].iloc[i]))
    A_rgb = torch.Tensor(ast.literal_eval(df["Color_A_RGB"].iloc[i]))
    B_rgb = torch.Tensor(ast.literal_eval(df["Color_B_RGB"].iloc[i]))
    preference = df["Preference"].iloc[i]

    anchor_hsl = torch.Tensor(convert_color(sRGBColor(*anchor_rgb),HSLColor).get_value_tuple())
    A_hsl = torch.Tensor(convert_color(sRGBColor(*A_rgb),HSLColor).get_value_tuple())
    B_hsl = torch.Tensor(convert_color(sRGBColor(*B_rgb),HSLColor).get_value_tuple())

    # if preference == "A":
    #     #positive pairing
    #     all_X.append(torch.concatenate([anchor_rgb,A_rgb]))
    #     all_y.append(torch.Tensor([1.]))

    #     #negative pairing
    #     all_X.append(torch.concatenate([anchor_rgb,B_rgb]))
    #     all_y.append(torch.Tensor([0.]))

    # elif preference == "B":
    #     #positive pairing
    #     all_X.append(torch.concatenate([anchor_rgb,B_rgb]))
    #     all_y.append(torch.Tensor([1.]))

    #     #negative pairing
    #     all_X.append(torch.concatenate([anchor_rgb,A_rgb]))
    #     all_y.append(torch.Tensor([0.]))

    if preference == "A":
        #positive pairing
        all_X.append(torch.concatenate([anchor_hsl,A_hsl]))
        all_y.append(torch.Tensor([1.]))
        all_X.append(torch.concatenate([A_hsl,anchor_hsl]))
        all_y.append(torch.Tensor([1.]))

        #negative pairing
        all_X.append(torch.concatenate([anchor_hsl,B_hsl]))
        all_y.append(torch.Tensor([0.]))
        all_X.append(torch.concatenate([B_hsl,anchor_hsl]))
        all_y.append(torch.Tensor([0.]))

    elif preference == "B":
        #positive pairing
        all_X.append(torch.concatenate([anchor_hsl,B_hsl]))
        all_y.append(torch.Tensor([1.]))
        all_X.append(torch.concatenate([B_hsl,anchor_hsl]))
        all_y.append(torch.Tensor([1.]))

        #negative pairing
        all_X.append(torch.concatenate([anchor_hsl,A_hsl]))
        all_y.append(torch.Tensor([0.]))
        all_X.append(torch.concatenate([A_hsl,anchor_hsl]))
        all_y.append(torch.Tensor([0.]))
    

# all_X = torch.stack(all_X,dim=0) * 2 - 1        # normalize to -1, 1
# all_y = torch.stack(all_y,dim=0) 

all_X = torch.stack(all_X,dim=0) * torch.Tensor([1/180,2,2,1/180,2,2]) - 1        # normalize to -1, 1
all_y = torch.stack(all_y,dim=0) 

#split dataset
all_idx = torch.randperm(all_X.shape[0])
train_fraction = 0.9
train_idx = all_idx[:int(train_fraction*all_X.shape[0])]
test_idx = all_idx[int(train_fraction*all_X.shape[0]):]

train_X = all_X[train_idx].to(device)
train_y = all_y[train_idx].to(device)


test_X = all_X[test_idx].to(device)
test_y = all_y[test_idx].to(device)


model = Encoder().to(device)
optim = torch.optim.Adam(model.parameters(),lr=0.0001)
criterion = torch.nn.BCELoss()

n_epochs = 1000000

train_X_split = torch.chunk(train_X,128,dim=0)
train_y_split = torch.chunk(train_y,128,dim=0)
test_X_split = torch.chunk(test_X,128,dim=0)
test_y_split = torch.chunk(test_y,128,dim=0)

for e in range(n_epochs):

    optim.zero_grad()
    y_hat = model(train_X)
    train_loss = criterion(y_hat,train_y)

    train_loss.backward()
    optim.step()

   
    if (e%1)==0:
        with torch.no_grad():
            y_hat_test = model(test_X)
            test_loss = criterion(y_hat_test,test_y)

            train_acc = (train_y.cpu() == (y_hat.cpu().detach()>=0.5)).to(torch.float64).mean().item()
            test_acc = (test_y.cpu() == (y_hat_test.cpu().detach()>=0.5)).to(torch.float64).mean().item()
        # print(f"{e}/{n_epochs} - Train Loss: {train_loss.detach().item()} - APR Train: {ap_fraction_train} - ANR Train: {an_fraction_train} - Test Loss: {test_loss.detach().item()} - APR Test: {ap_fraction_test} - ANR Test: {an_fraction_test}")
        print(f"{e}/{n_epochs} - Train Loss: {train_loss.cpu().detach().item()} - Train Acc: {train_acc} - Test Loss: {test_loss.cpu().detach().item()} - Test Acc: {test_acc}")


print("stop")
