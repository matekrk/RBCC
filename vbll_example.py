import sys, os
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import vbll_classifier
from sklearn.metrics import zero_one_loss, hamming_loss, f1_score, accuracy_score
import torch

from utils import *

VBLL_PATH_UPTODATE = "/home/pyla/bayesian/vbll_uptodate/vbll"
sys.path.append(os.path.abspath(VBLL_PATH_UPTODATE))
import vbll

def setup_data(classes, input_features):
    train_data, train_labels = load_scene(partition='Train')
    train_labels = make_binary(train_labels, classes)

    X_train = train_data[input_features].values
    y_train = train_labels[classes].values

    X_train = X_train[y_train.sum(axis=1) != 0]
    y_train = y_train[y_train.sum(axis=1) != 0]

    test_data, test_labels = load_scene(partition='Test')
    test_labels = make_binary(test_labels, classes)

    X_test = test_data[input_features].values
    y_test = test_labels[classes].values

    X_test = X_test[y_test.sum(axis=1) != 0]
    y_test = y_test[y_test.sum(axis=1) != 0]

    return X_train, y_train, X_test, y_test

def setup_hyperparameters(len_data):
    batch_size = 128
    num_epochs = 1000
    num_runs = 3
    learning_rate = 1e-3
    reg_weight = 1. / len_data
    param = 'diagonal' # diagonal, lowrank, dense
    softmax_bound = 'jensen'
    return_ood = False
    prior_scale = 1.0
    noise_label = True

    hidden_size = 294
    n_hidden = 0

    return hidden_size, n_hidden, batch_size, num_epochs, num_runs, learning_rate, reg_weight, param, softmax_bound, return_ood, prior_scale, noise_label

def do_experiments():
    exp_name = 'scene'
    classes = ['Beach', 'Sunset', 'FallFoliage', 'Field', 'Mountain', 'Urban']
    num_input_features = 294
    input_features = ['Att' + str(i) for i in range(1, num_input_features+1)]

    X_train, y_train, X_test, y_test = setup_data(classes, input_features)
    num_classes = len(classes)
    len_data = len(X_train)
    hidden_size, n_hidden, batch_size, num_epochs, num_runs, learning_rate, reg_weight, param, softmax_bound, return_ood, prior_scale, noise_label = setup_hyperparameters(len_data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    verbose = False

    whole_exp_name = exp_name + f"_vbll_extended_results_{hidden_size}x{n_hidden}_{param}_{prior_scale}_{learning_rate}_{num_epochs}_cuda"

    vbll_clf_dict = {}
    fig, ax1 = plt.subplots(figsize=(10, 10))
    ax2 = ax1.twinx()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax2.set_ylabel('Accuracy', color='tab:orange')
    base_blue = to_rgba('tab:blue')
    base_orange = to_rgba('tab:orange')

    for run in range(num_runs):
        vbll_clf_dict[run] = {}

        print(f'run {run}')
        clf = vbll_classifier.ExtendedVBLLClassifierClf(num_input_features, hidden_size, n_hidden, batch_size, classes, reg_weight, param, softmax_bound, return_ood, prior_scale, noise_label, device, learning_rate, num_epochs, verbose)

        loss_list, acc_list = clf.fit(X_train, y_train)

        blue_color = (base_blue[0], base_blue[1], base_blue[2], 1 - run * 0.1)
        orange_color = (base_orange[0], base_orange[1], base_orange[2], 1 - run * 0.1)        
        ax1.plot(loss_list, label='Loss run ' + str(run), color=blue_color)
        ax2.plot(acc_list, label='Accuracy run ' + str(run), color=orange_color)
        
        clf.save(f'./{whole_exp_name}_{run}.pt')

        preds, y_pred = clf.predict_with_proba(X_test)
        vbll_clf_dict[run]['y_test'] = y_test
        vbll_clf_dict[run]['y_pred'] = y_pred
        vbll_clf_dict[run]['y_pred_proba'] = preds

        vbll_clf_dict[run]['01_loss'] = zero_one_loss(y_test, y_pred)
        print('[1 minus] 0-1 Accuracy [=SA]', 1. - vbll_clf_dict[run]['01_loss'])
        vbll_clf_dict[run]['hamming_loss'] = hamming_loss(y_test, y_pred)
        print('Hamming Loss', vbll_clf_dict[run]['hamming_loss'])
        for i in range(num_classes):
            vbll_clf_dict[run][f'f1_score_{i}'] = f1_score(y_test[:, i], y_pred[:, i])
            print(f'F1 score for class {i}', vbll_clf_dict[run][f'f1_score_{i}'])
            vbll_clf_dict[run][f'accuracy_{i}'] = accuracy_score(y_test[:, i], y_pred[:, i])
            print(f'Accuracy for class {i}:', vbll_clf_dict[run][f'accuracy_{i}'])        

    plt.legend()
    plt.show()
    plt.savefig(f'./{whole_exp_name}.png')
    with open(whole_exp_name + ".pkl", "wb") as f:
        pickle.dump(vbll_clf_dict, f)

def main():
    do_experiments()

if __name__ == "__main__":
    main()
