import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

# Lire le dataset avec l'entête 'valeurs'
dataset = pd.read_csv('datasetLSTM.csv')

# Ajouter une colonne "month"
dataset['month'] = dataset.index.map(lambda x: f"1949-{x+1:02d}")

# Réorganiser les colonnes
dataset = dataset[['month', 'nbrconc']]

# Enregistrer le dataset modifié
dataset.to_csv('dataLSTM.csv', index=False)

"""## Data Plot"""

# Fixer la graine aléatoire pour numpy
np.random.seed(998)

training_set = pd.read_csv('dataLSTM.csv')
#training_set = pd.read_csv('shampoo.csv')

training_set = training_set.iloc[:,1:2].values

#plt.plot(training_set, label = 'Shampoo Sales Data')
plt.plot(training_set, label = 'Data')
plt.show()

"""## Dataloading"""

def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

sc = MinMaxScaler()
training_data = sc.fit_transform(training_set)

seq_length = 4
x, y = sliding_windows(training_data, seq_length)

train_size = int(len(y) * 0.67)
test_size = len(y) - train_size

# train_size = 17156
# test_size = 120

dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))

trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

print(train_size)
print(test_size)

"""## Model & Trainig

## Code original

Input Size : 1 (Ce sont les dimensions de l'entrée à chaque pas de temps)
Hidden Size : 2 (Le nombre de neurones dans la couche cachée de l'unité LSTM)
Num Layers : 1 (Le nombre de couches cachées dans l'unité LSTM)
Output Size (Num Classes) : 1 (La dimension de sortie du réseau)
Fonction de perte : Mean Squared Error (MSE), utilisée pour la régression
Optimiseur : Adam avec un taux d'apprentissage de 0.01
Nombre d'époques : 2000
Le modèle comporte une seule couche cachée LSTM avec 2 neurones, une couche de sortie linéaire pour une classe de sortie, et il est entraîné pour minimiser l'erreur quadratique moyenne (MSE) sur 2000 époques.
"""

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out


num_epochs = 2000
learning_rate = 0.01

input_size = 1
hidden_size = 2
num_layers = 1

num_classes = 1

lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    outputs = lstm(trainX)
    optimizer.zero_grad()

    # obtain the loss function
    loss = criterion(outputs, trainY)

    loss.backward()

    optimizer.step()
    if epoch % 100 == 0:
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

"""## Code ML

LSTM model has 3 layers as shown in Fig 5. LSTM
layer with 64 units, which takes the input sequences. Dense
layer with 32 units, which applies a linear transformation
to the output of the LSTM layer. This layer utilizes ReLU
activation function. Dropout layer with a rate of 0.2, which
applies regularization by randomly setting a fraction of the
inputs to 0. Dense layer with 1 unit, which produces the fnal
output prediction.
"""

# import torch
# import torch.nn as nn
# from torch.autograd import Variable

# class LSTM(nn.Module):

#     def __init__(self, input_size, hidden_size, num_layers):
#         super(LSTM, self).__init__()

#         self.num_layers = num_layers
#         self.hidden_size = hidden_size

#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
#                             num_layers=num_layers, batch_first=True)

#         self.fc1 = nn.Linear(hidden_size, 32)  # Dense layer with 32 units
#         self.relu = nn.ReLU()                  # ReLU activation function
#         self.dropout = nn.Dropout(p=0.2)       # Dropout layer with rate of 0.2
#         self.fc2 = nn.Linear(32, 1)            # Dense layer with 1 unit for final output

#     def forward(self, x):
#         h_0 = Variable(torch.zeros(
#             self.num_layers, x.size(0), self.hidden_size))

#         c_0 = Variable(torch.zeros(
#             self.num_layers, x.size(0), self.hidden_size))

#         # Propagate input through LSTM
#         _, (h_out, _) = self.lstm(x, (h_0, c_0))

#         h_out = h_out[-1, :, :]  # Taking only the last layer's output
#         out = self.fc1(h_out)
#         out = self.relu(out)
#         out = self.dropout(out)
#         out = self.fc2(out)

#         return out


# # Define the model with specified configuration
# input_size = 1
# hidden_size = 64
# num_layers = 3

# lstm = LSTM(input_size, hidden_size, num_layers)

# # Define the loss function and optimizer
# criterion = torch.nn.MSELoss()    # mean-squared error for regression
# optimizer = torch.optim.Adam(lstm.parameters(), lr=0.01)

# # Assuming trainX and trainY are already defined with appropriate data

# # Train the model
# num_epochs = 51
# for epoch in range(num_epochs):
#     outputs = lstm(trainX)
#     optimizer.zero_grad()

#     # obtain the loss function
#     loss = criterion(outputs, trainY)

#     loss.backward()

#     optimizer.step()
#     print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

"""## Code Vahi

The LSTM network has
five hidden layers, each with 32 neurons and a ReLU
activation function. In addition, to prevent overfitting, a
Dropout layer with a value of 0.5 is used. The network
also has one output and a linear activations function. The
mean squared error (MSE) is also used as a loss function.
"""

# import torch
# import torch.nn as nn

# class LSTM(nn.Module):

#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(LSTM, self).__init__()

#         self.num_layers = num_layers
#         self.hidden_size = hidden_size

#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
#                             num_layers=num_layers, batch_first=True)

#         self.dropout = nn.Dropout(0.5)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

#         ula, (h_out, _) = self.lstm(x, (h_0, c_0))

#         h_out = self.dropout(h_out[-1])  # Take the last layer's hidden state and apply dropout
#         out = self.fc(h_out)

#         return out

# # Configuration
# input_size = 1
# hidden_size = 32
# num_layers = 5
# output_size = 1

# # Initialize LSTM
# lstm = LSTM(input_size, hidden_size, num_layers, output_size)

# # Loss and optimizer
# criterion = nn.MSELoss()    # Mean Squared Error loss
# optimizer = torch.optim.Adam(lstm.parameters(), lr=0.01)

# # Training loop
# for epoch in range(51):
#     outputs = lstm(trainX)
#     optimizer.zero_grad()
#     loss = criterion(outputs, trainY)
#     loss.backward()
#     optimizer.step()
#     print("Epoch: %d, Loss: %.5f" % (epoch, loss.item()))

"""## Testing """

import pandas as pd

lstm.eval()
train_predict = lstm(dataX)

data_predict = train_predict.data.numpy()
dataY_plot = dataY.data.numpy()

data_predict = sc.inverse_transform(data_predict)
dataY_plot = sc.inverse_transform(dataY_plot)

# Convertir les données en DataFrame
df = pd.DataFrame({'Valeurs réelles': dataY_plot.flatten(), 'Valeurs prédites': data_predict.flatten()})

# Enregistrer le DataFrame dans un fichier Excel
df[['Valeurs réelles', 'Valeurs prédites']].to_excel('predictionsall.xlsx', index=False)

# Afficher le graphique
plt.axvline(x=train_size, c='r', linestyle='--')
plt.plot(dataY_plot, label='Valeurs réelles')
plt.plot(data_predict, label='Valeurs prédites')
plt.suptitle('Prédiction de série temporelle')
plt.legend()
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
from scipy.stats import spearmanr
import numpy as np

# Calculer MASE
def calculate_mase(y_true, y_pred, y_train):
    mae = mean_absolute_error(y_true, y_pred)
    naive_mae = np.mean(np.abs(y_train[1:] - y_train[:-1]))
    return mae / naive_mae

# Calculer sMAPE
def calculate_smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff) * 100

# Calculer la racine carrée de l'erreur quadratique moyenne (RMSE)
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Calculer le RMSE normalisé
def calculate_normalized_rmse(y_true, y_pred):
    return calculate_rmse(y_true, y_pred) / (np.max(y_true) - np.min(y_true))

# Calculer le coefficient de détermination (R2 Score)
def calculate_r2_score(y_true, y_pred):
    return r2_score(y_true, y_pred)

# Calculer la corrélation de Spearman
def calculate_spearman_corr(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]

# Calculer la variance expliquée
def calculate_explained_variance(y_true, y_pred):
    return explained_variance_score(y_true, y_pred)

# Calculer les métriques
mase = calculate_mase(dataY_plot, data_predict, dataY_plot[:train_size])
smape = calculate_smape(dataY_plot, data_predict)
rmse = calculate_rmse(dataY_plot, data_predict)
normalized_rmse = calculate_normalized_rmse(dataY_plot, data_predict)
r2 = calculate_r2_score(dataY_plot, data_predict)
spearman_corr = calculate_spearman_corr(dataY_plot, data_predict)
explained_var = calculate_explained_variance(dataY_plot, data_predict)

# Afficher les résultats
print("MASE:", mase)
print("sMAPE:", smape)
print("RMSE:", rmse)
print("Normalized RMSE:", normalized_rmse)
print("R2 Score:", r2)
print("Spearman Correlation:", spearman_corr)
print("Explained Variance:", explained_var)

"""# 120"""

# import pandas as pd

# lstm.eval()
# train_predict = lstm(dataX)

# data_predict = train_predict.data.numpy()
# dataY_plot = dataY.data.numpy()

# data_predict = sc.inverse_transform(data_predict)
# dataY_plot = sc.inverse_transform(dataY_plot)

# # Sélectionner les 120 dernières valeurs
# data_predict_last_120 = data_predict[-120:]
# dataY_plot_last_120 = dataY_plot[-120:]

# # Convertir les données en DataFrame
# df = pd.DataFrame({'Valeurs réelles': dataY_plot_last_120.flatten(), 'Valeurs prédites': data_predict_last_120.flatten()})

# # Enregistrer le DataFrame dans un fichier Excel
# df[['Valeurs réelles', 'Valeurs prédites']].to_excel('predictions120.xlsx', index=False)

# # Afficher le graphique
# plt.axvline(x=train_size, c='r', linestyle='--')
# plt.plot(dataY_plot_last_120, label='Valeurs réelles')
# plt.plot(data_predict_last_120, label='Valeurs prédites')
# plt.suptitle('Prédiction de série temporelle')
# plt.legend()
# plt.show()

# from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
# from scipy.stats import spearmanr
# import numpy as np

# # Calculer MASE
# def calculate_mase(y_true, y_pred, y_train):
#     mae = mean_absolute_error(y_true, y_pred)
#     naive_mae = np.mean(np.abs(y_train[1:] - y_train[:-1]))
#     return mae / naive_mae

# # Calculer sMAPE
# def calculate_smape(y_true, y_pred):
#     denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
#     diff = np.abs(y_true - y_pred) / denominator
#     diff[denominator == 0] = 0.0
#     return np.mean(diff)

# # Calculer la racine carrée de l'erreur quadratique moyenne (RMSE)
# def calculate_rmse(y_true, y_pred):
#     return np.sqrt(mean_squared_error(y_true, y_pred))

# # Calculer le RMSE normalisé
# def calculate_normalized_rmse(y_true, y_pred):
#     return calculate_rmse(y_true, y_pred) / (np.max(y_true) - np.min(y_true))

# # Calculer le coefficient de détermination (R2 Score)
# def calculate_r2_score(y_true, y_pred):
#     return r2_score(y_true, y_pred)

# # Calculer la corrélation de Spearman
# def calculate_spearman_corr(y_true, y_pred):
#     return spearmanr(y_true, y_pred)[0]

# # Calculer la variance expliquée
# def calculate_explained_variance(y_true, y_pred):
#     return explained_variance_score(y_true, y_pred)

# # Sélectionner les 120 dernières valeurs de dataY_plot et data_predict
# dataY_plot_last_120 = dataY_plot[-120:]
# data_predict_last_120 = data_predict[-120:]

# # Calculer les métriques
# mase = calculate_mase(dataY_plot_last_120, data_predict_last_120, dataY_plot_last_120[:train_size])
# smape = calculate_smape(dataY_plot_last_120, data_predict_last_120)
# rmse = calculate_rmse(dataY_plot_last_120, data_predict_last_120)
# normalized_rmse = calculate_normalized_rmse(dataY_plot_last_120, data_predict_last_120)
# r2 = calculate_r2_score(dataY_plot_last_120, data_predict_last_120)
# spearman_corr = calculate_spearman_corr(dataY_plot_last_120, data_predict_last_120)
# explained_var = calculate_explained_variance(dataY_plot_last_120, data_predict_last_120)

# # Afficher les résultats
# print("MASE:", mase)
# print("sMAPE:", smape)
# print("Explained Variance:", explained_var)
# print("RMSE:", rmse)
# print("Normalized RMSE:", normalized_rmse)
# print("R2 Score:", r2)
# print("Spearman Correlation:", spearman_corr)