import torch 
from sklearn.base import BaseEstimator
import models
import numpy as np
from tqdm import tqdm

class SoftmaxClassifierClf(BaseEstimator): # Inherits scikit-learn base classifier
    def __init__(self, input_size, batch_size, classes, noise_label,
                 device='cuda', learning_rate = 1e-2, num_epochs = 200, verbose=False):
        
        self.device = device
        output_heads = 2
        self.n_heads = len(classes)
        self.model_heads = [models.SoftmaxClassifier(input_size, output_heads).to(self.device) for _ in range(self.n_heads)]
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.classes = classes

    def forward(self, X):
        return [model(X) for model in self.model_heads]
    
    def forward_preds(self, X):
        preds = []
        for model in self.model_heads:
            out = model(X)
            preds.append(out)
        return torch.stack(preds, dim=1)
    
    def forward_predict(self, X):
        preds = self.forward_preds(X)
        return preds, torch.argmax(preds, dim=-1)
    
    def eval_acc_head(self, preds, y):
        map_preds = torch.argmax(preds, dim=1)
        return (map_preds == y).float().mean()
    
    def fit(self, X, y):
        for model in self.model_heads:
            model.train()

        self.optimizer = torch.optim.Adam(
            [param for model in self.model_heads for param in model.parameters()],
            lr=self.learning_rate
        )
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), 
                                                       torch.tensor(y, dtype=torch.float32))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)


        # --- train the model ---
        total_step = len(train_loader)
        loss_list = []
        acc_list = []
        for epoch in tqdm(range(self.num_epochs)):
            final_loss = 0.0
            acc_epoch = 0.0
            for i, (X, y) in enumerate(train_loader):
                X = X.to(self.device)
                y = y.to(self.device)
                y = y.float()

                # --- Forward pass ---
                # --- Backward and optimize ---
                iter_sum_loss, iter_losses = torch.tensor(0.0).to(self.device), []
                for i, model in enumerate(self.model_heads):
                    out = model(X)
                    logits = out
                    y_head = y[:, i].long()
                    acc = self.eval_acc_head(probs, y_head).item()
                    acc_epoch += acc * X.size(0) / len(self.model_heads)
                    log_softmax = torch.nn.functional.log_softmax(logits, dim=1)
                    loss = torch.nn.functional.nll_loss(log_softmax, y_head)
                    iter_sum_loss += loss
                    iter_losses.append(loss)

                self.optimizer.zero_grad()
                final_loss += iter_sum_loss.item()
                iter_sum_loss.backward()
                self.optimizer.step()
                if self.verbose:
                    if (i+1) % 1 == 0:
                        probs = out
                        acc = self.eval_acc(probs, y).item()
                        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}' .format(epoch+1, self.num_epochs, i+1, total_step, loss.item(), acc.item()))
            loss_list.append(final_loss/float(i))
            acc_list.append(acc_epoch/len(train_dataset))
            lr_scheduler.step()

        return loss_list, acc_list
    
    def predict_with_proba(self, X):
        for model in self.model_heads:
            model.eval()
        test_dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            preds = []
            y_preds = []
            for X in test_loader:
                X = X[0].to(self.device)

                batch_preds, batch_y_preds = self.forward_predict(X)

                preds.append(batch_preds.cpu().numpy())
                y_preds.append(batch_y_preds.cpu().numpy())

            preds = np.concatenate(preds, axis=0)
            y_preds = np.concatenate(y_preds, axis=0)

        return preds, y_preds
    
    def save(self, path):
        torch.save(self.model_heads, path)

    def load(self, path, device=torch.device('cpu')):
        self.model_heads = torch.load(path, map_location=device)
        for model in self.model_heads:
            model.to(self.device)

class ExtendedSoftmaxClassifierClf(BaseEstimator): # Inherits scikit-learn base classifier
    def __init__(self, input_size, hidden_size, n_hidden, batch_size, classes, noise_label,
                 device='cuda', learning_rate = 1e-2, num_epochs = 200, verbose=False):
        
        self.device = device
        self.model_base = None
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        if hidden_size or n_hidden:
            self.model_base = models.SimpleBaseModel(input_size, hidden_size, n_hidden).to(self.device)
        output_heads = 2
        self.n_heads = len(classes)
        self.model_heads = [models.SoftmaxClassifier(input_size, output_heads).to(self.device) for _ in range(self.n_heads)]
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.classes = classes

    def forward(self, X):
        if self.model_base is not None:
            X = self.model_base(X)
        return [model(X) for model in self.model_heads]
    
    def forward_preds(self, X):
        if self.model_base is not None:
            X = self.model_base(X)
        preds = []
        for model in self.model_heads:
            out = model(X)
            preds.append(out)
        return torch.stack(preds, dim=1)
    
    def forward_predict(self, X):
        preds = self.forward_preds(X)
        return preds, torch.argmax(preds, dim=-1)
    
    def eval_acc_head(self, preds, y):
        map_preds = torch.argmax(preds, dim=1)
        return (map_preds == y).float().mean()
    
    def fit(self, X, y):
        self.model_base.train()
        for model in self.model_heads:
            model.train()

        learnable_parameters = [param for model in self.model_heads for param in model.parameters()]
        if self.model_base is not None:
            learnable_parameters += [param for param in self.model_base.parameters()]
        print("Number of (learnable) parameters in the model: ", sum(p.numel() for p in learnable_parameters if p.requires_grad))

        self.optimizer = torch.optim.Adam(learnable_parameters,
            lr=self.learning_rate
        )
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), 
                                                       torch.tensor(y, dtype=torch.float32))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # --- train the model ---
        total_step = len(train_loader)
        loss_list = []
        acc_list = []
        for epoch in tqdm(range(self.num_epochs)):
            final_loss = 0.0
            acc_epoch = 0.0
            for i, (X, y) in enumerate(train_loader):
                X = X.to(self.device)
                y = y.to(self.device)
                y = y.float()

                if self.model_base is not None:
                    X = self.model_base(X)

                # --- Forward pass ---
                # --- Backward and optimize ---
                iter_sum_loss, iter_losses = torch.tensor(0.0).to(self.device), []
                for i, model in enumerate(self.model_heads):
                    out = model(X)
                    logits = out
                    y_head = y[:, i].long()
                    acc = self.eval_acc_head(logits, y_head).item()
                    acc_epoch += acc * X.size(0) / len(self.model_heads)
                    log_softmax = torch.nn.functional.log_softmax(logits, dim=1)
                    loss = torch.nn.functional.nll_loss(log_softmax, y_head)
                    iter_sum_loss += loss
                    iter_losses.append(loss)

                self.optimizer.zero_grad()
                final_loss += iter_sum_loss.item()
                iter_sum_loss.backward()
                self.optimizer.step()
                if self.verbose:
                    if (i+1) % 1 == 0:
                        probs = out
                        acc = self.eval_acc(probs, y).item()
                        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}' .format(epoch+1, self.num_epochs, i+1, total_step, loss.item(), acc.item()))
            loss_list.append(final_loss/float(i))
            acc_list.append(acc_epoch/len(train_dataset))
            lr_scheduler.step()

        return loss_list, acc_list
    
    def predict_with_proba(self, X):
        for model in self.model_heads:
            model.eval()
        test_dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            preds = []
            y_preds = []
            for X in test_loader:
                X = X[0].to(self.device)

                batch_preds, batch_y_preds = self.forward_predict(X)

                preds.append(batch_preds.cpu().numpy())
                y_preds.append(batch_y_preds.cpu().numpy())

            preds = np.concatenate(preds, axis=0)
            y_preds = np.concatenate(y_preds, axis=0)

        return preds, y_preds
    
    def save(self, path):
        if self.model_base is not None:
            torch.save((self.model_base, self.model_heads), path)
        else:
            torch.save(self.model_heads, path)

    def load(self, path, device=torch.device('cpu')):
        if self.model_base is not None:
            self.model_base, self.model_heads = torch.load(path, map_location=device)
        else:
            self.model_heads = torch.load(path, map_location=device)
        for model in self.model_heads:
            model.to(self.device)
        if self.model_base is not None:
            self.model_base.to(self.device)
