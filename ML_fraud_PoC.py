##############################################
#           Imports de modules               #
##############################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from sklearn.preprocessing import StandardScaler

##############################################
#                 Réglages                   #
##############################################

data = np.loadtxt('output.csv', delimiter=",", dtype=np.float32, skiprows=1)

input_params = 5
hidder_layer = 15
outputs = 1

max_epochs = 10000


##############################################
#         Variables du programme             #
##############################################

data_tensor = torch.tensor(data)

train_set = data_tensor[:150]
x_train = train_set[:,:-1]
y_train = train_set[:,-1].view(train_set.shape[0], -1)

test_set = data_tensor[150:]
x_test = test_set[:,:-1]
y_test = test_set[:,-1].view(test_set.shape[0], -1)

##############################################
#           Structure du réseau              #
##############################################

class Net(nn.Module):

    def __init__(self, _input_params, _hidden_layer, _outputs):
        super(Net, self).__init__()

        self.input_params = _input_params
        self.hidden_layer = _hidden_layer
        self.outputs = _outputs

        self.fc1 = nn.Linear(self.input_params, self.hidden_layer)
        self.fc2 = nn.Linear(self.hidden_layer, self.hidden_layer)
        self.fc3 = nn.Linear(self.hidden_layer, self.outputs)

    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

##############################################
#              Entrainement                  #
##############################################

# instanciation du réseau de neurones
model = Net(input_params, hidder_layer, outputs)

# critère d'optimisation
criterion = torch.nn.MSELoss()

# optimisateur
optimizer = optim.SGD(model.parameters(), lr=1e-2)

# passage en mode entrainement
model.train()

# répétition de l'entrainement 
for t in range(max_epochs):

    #estimation des risques de fraudes
    y_pred = model(x_train)

    # calcul de "l'erreur"
    loss = criterion(y_pred, y_train.reshape(-1,1))


    if t % 100 == 99:
        print(t, loss.item())

    #reset de l'erreur et propagation des corrections
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#print((y_pred>0.5)==y_train )


#estimation de l'erreur sur la base de test
y_pt = model(x_test)
print(float(sum((y_pt>0.5) == y_test)/test_set.shape[0]*100),"% de succès")