import torch 
import torch.nn as nn
import numpy as np
from utils import *

class SimpleBaseModel(nn.Module):
    def __init__(self, input_dim, hidden_size, n_hidden, leaky_relu_slope=0.0, device='cuda'):
        super(SimpleBaseModel, self).__init__()
        self.NAME = "SimpleBaseModel"
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.layers = []
        # be careful - if n_hidden is 0 then just a linear layer from input_dim to input_dim
        cur_dim = input_dim
        for i in range(self.n_hidden):
            self.layers.append(nn.Linear(cur_dim, hidden_size))
            cur_dim = hidden_size
        self.layers.append(nn.Linear(cur_dim, input_dim))
        self.layers = nn.ModuleList(self.layers)
        self.non_linearity = nn.LeakyReLU(leaky_relu_slope)
        self.device = device

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
            X = torch.relu(X)
        return X

class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, softmax = False, device='cuda'):
        super(SoftmaxClassifier, self).__init__()
        self.NAME = "SoftmaxClassifier"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.out = nn.Linear(input_dim, self.output_dim)
        self.softmax = softmax

    def forward(self, X):
        out = self.out(X)
        if self.softmax:
            return torch.nn.functional.softmax(out, dim=1)
        return out
    
####### FFN 

class FFN(nn.Module):
    def __init__(self, INPUT_DIM, HIDDEN_DIM, DEVICE='cpu'):
        super(FFN, self).__init__()
        self.NAME = "FFN"
        self.HIDDEN_DIM = HIDDEN_DIM
        self.INPUT_DIM = INPUT_DIM
        self.devivce = DEVICE

        self.input = nn.Linear(self.INPUT_DIM, self.HIDDEN_DIM).to(self.devivce)
        self.out = nn.Linear(self.HIDDEN_DIM, 1)

    def forward(self, X):
        X1 = torch.sigmoid(self.input(X))
        X2 = self.out(X1)

        y_hat = X2.view(-1)

        return y_hat

    def computeLoss(self, logits, labels):
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logits, labels)
        return loss

class bayesian_classifier_chain_clf(nn.Module):
    def __init__(self, INPUT_DIM, HIDDEN_DIM, N_CLASSES, PARENT_DICT,  DEVICE='cuda', CELL_TYPE='GRU'):
        super(bayesian_classifier_chain_clf, self).__init__()
        self.NAME = "bayesian_classifier_chain_clf"
        self.HIDDEN_DIM = HIDDEN_DIM
        self.INPUT_DIM = INPUT_DIM
        self.N_CLASSES = N_CLASSES
        self.CELL_TYPE = CELL_TYPE
        self.device = DEVICE
        self.N_LAYERS = 1
        self.parent_dict = PARENT_DICT

        if CELL_TYPE == "RNN":
            self.rnn_cell = nn.RNN(INPUT_DIM+self.N_CLASSES, HIDDEN_DIM, self.N_LAYERS, batch_first=False) 
        elif CELL_TYPE == "GRU":
            self.rnn_cell = nn.GRU(INPUT_DIM+self.N_CLASSES, HIDDEN_DIM, self.N_LAYERS, batch_first=False) 
        elif CELL_TYPE == "LSTM":
            self.rnn_cell = nn.LSTM(INPUT_DIM+self.N_CLASSES, HIDDEN_DIM, self.N_LAYERS, batch_first=False) 

        self.out = nn.Linear(HIDDEN_DIM, self.N_CLASSES) 

    def initHidden(self, BATCH_SIZE):
        if self.CELL_TYPE == "LSTM":
            return (torch.zeros(self.N_LAYERS, BATCH_SIZE, self.HIDDEN_DIM).to(self.device),
                    torch.zeros(self.N_LAYERS, BATCH_SIZE, self.HIDDEN_DIM).to(self.device))
        else:
            return torch.zeros(self.N_LAYERS, BATCH_SIZE, self.HIDDEN_DIM).to(self.device)

    def forward(self, X, Y=False, TRAINING = True):


        predictions = []

        prev_preds = torch.Tensor(np.zeros(X.shape[1])).to(self.device)

        X_aug = torch.cat((prev_preds.unsqueeze(0).unsqueeze(2), X), dim=2)

        if TRAINING:
            pred_list = []
            for instance in range(X.shape[1]):
                x = X[:,instance,:].unsqueeze(1)
                #print('Y shape', Y.shape)
                y = Y[instance] #if Y is [b, C]
                #print('y shape', y.shape)
                
                pred_y = torch.Tensor(np.zeros(self.N_CLASSES)).to(self.device)
                for c in range(self.N_CLASSES):
                    self.state = self.initHidden(x.shape[1])
                    parents = self.parent_dict[c]['parents']
                    if len(parents) == 0:
                        parent_vec = torch.Tensor(np.zeros(self.N_CLASSES)).to(self.device)
                        x_aug = torch.cat((parent_vec.unsqueeze(0).unsqueeze(1), x), dim=2)
                        output, self.state = self.rnn_cell(x_aug, self.state)
                        pred_y[c] += self.out(output).flatten()[c] 
                        
                    else:
                        for p in parents:
                            parent_vec = torch.Tensor(np.zeros(self.N_CLASSES)).to(self.device)
                            parent_vec[p] = y[p] 
                            x_aug = torch.cat((parent_vec.unsqueeze(0).unsqueeze(1), x), dim=2)
                            output, self.state = self.rnn_cell(x_aug, self.state)
                        pred_y[c] += self.out(output).flatten()[c] 
                pred_y = pred_y.view(1,-1) 
                pred_list.append(pred_y)
            y_pred = torch.cat(pred_list)
            return y_pred
        else:
            predicted_dict = {}
            code = 0
            for instance in range(X.shape[1]):
                x = X[:,instance,:].unsqueeze(1)
                x_k = 'x' + str(code)
                predicted_dict[x_k] = {}
                for c in range(self.N_CLASSES):
                    self.infer(x, x_k, c, predicted_dict)
                code += 1
            return predicted_dict
    
    def infer(self, x, x_k, c, predicted_dict):
        parents = self.parent_dict[c]['parents']
        state = self.initHidden(x.shape[1])
        if c not in predicted_dict[x_k].keys():
            if len(parents) == 0:
                parent_vec = torch.Tensor(np.zeros(self.N_CLASSES)).to(self.device)
                x_aug = torch.cat((parent_vec.unsqueeze(0).unsqueeze(1), x), dim=2)
                output, state = self.rnn_cell(x_aug, state)
                predicted_dict[x_k][c] = torch.sigmoid(self.out(output).flatten()[c])
            else:
                for p in parents:
                    if p not in predicted_dict[x_k].keys():
                        self.infer(x,x_k,p,predicted_dict)
                state = self.initHidden(x.shape[1])
                for p in parents:
                    parent_vec = torch.Tensor(np.zeros(self.N_CLASSES)).to(self.device)
                    parent_vec[p] = torch.round(predicted_dict[x_k][p])
                    x_aug = torch.cat((parent_vec.unsqueeze(0).unsqueeze(1), x), dim=2)
                    output, state = self.rnn_cell(x_aug, state)
                predicted_dict[x_k][c] = torch.sigmoid(self.out(output).flatten()[c]) #if out is [c]

    
    def computeLoss(self, logits, labels):
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logits, labels)
        return loss
