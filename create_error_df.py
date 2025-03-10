# --- Device configuration ---
import math
import numpy as np
import pandas as pd
import torch
from skmultilearn.problem_transform import BinaryRelevance
from your_module import FfnClf  # Replace 'your_module' with the actual module name where FfnClf is defined

# --- Hyper-parameters ---
input_size = len(input_features)
hidden_size = 100
n_hidden = 1
num_classes = len(classes)
batch_size = 128
num_epochs = 200
learning_rate = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train_df = train_data[input_features]
train_ls = train_labels[classes]

cutoff = math.floor(train_df.shape[0]/2.)
first_half_data = train_df[:cutoff]
last_half_data = train_df[cutoff:]
first_half_ls = train_ls[:cutoff]
last_half_ls = train_ls[cutoff:]

clf = BinaryRelevance(classifier=FfnClf(input_size, hidden_size, num_classes, batch_size, device, learning_rate, num_epochs, verbose=False))
clf.fit(first_half_data, first_half_ls)

preds_last_half = clf.predict(last_half_data.values)

clf = BinaryRelevance(classifier=FfnClf(input_size, hidden_size, num_classes, batch_size, device, learning_rate, num_epochs, verbose=False))
clf.fit(last_half_data, last_half_ls)

preds_first_half = clf.predict(first_half_data.values)

preds_first = np.asarray(preds_first_half.todense())
preds_last = np.asarray(preds_last_half.todense())
preds_combined = np.vstack((preds_first, preds_last))
errors = train_ls.values - preds_combined
error_df = pd.DataFrame(errors, columns=classes)
