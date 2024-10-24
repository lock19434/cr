import os
from pyexpat import model
import random
from tqdm import tqdm
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
import pandas as pd
from numpy import vstack
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.optim import SGD
from torch.nn import BCELoss
import json
from utils.quantize_utils import aggregated_quantize_encrypt, aggregated_encrypt
from utils.options import args_parser
from sklearn.metrics import roc_curve, auc

# get args
args = args_parser()
print(args)


class PeopleDataset(T.utils.data.Dataset):
    def __init__(self, src_file, num_rows=None):
        df = pd.read_csv(src_file)
        df.drop(df.columns[[0]], axis=1, inplace=True)
        print(df.columns)
        df.Class = df.Class.astype('float64')
        y_tmp = df['Class'].values
        x_tmp = df.drop('Class', axis=1).values

        self.x_data = T.tensor(x_tmp, dtype=T.float64).to(device)
        self.y_data = T.tensor(y_tmp, dtype=T.float64).to(device)

        print(type(self.x_data))
        print(len(self.x_data))

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if T.is_tensor(idx):
            idx = idx.tolist()
        preds = self.x_data[idx].type(T.FloatTensor)
        pol = self.y_data[idx].type(T.LongTensor)
        sample = [preds, pol]
        return sample


class MLPUpdated(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(30, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

    fed_acc, fed_pre, fed_recall, fed_f1 = list(), list(), list(), list()


def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig("no_e_roc.png")
    plt.close()


mp = {}

# Can be changed to 12,16,24,32
num_clients = args.num_clients
# Change it to 3, 6, 10, 16
# cln = max(int(args.frac * args.num_clients), 1)
cln = args.num_select
num_selected = cln
num_rounds = args.num_rounds
epochs = 5
batch_size = args.batch_size
device = "cuda:0"
device = T.device(device)
fed_acc, fed_pre, fed_recall, fed_f1 = list(), list(), list(), list()

# Dividing the training data into num_clients, with each client having equal number of data
# traindata = PeopleDataset('./creditcard_train_SMOTE_1.csv')
traindata = PeopleDataset('./' + args.dataset)
print(len(traindata))
base_split = len(traindata) // num_clients
remainder = len(traindata) % num_clients
splits = [base_split] * num_clients
for i in range(remainder):
    splits[i] += 1
traindata_split = T.utils.data.random_split(traindata, splits)

train_loader = [T.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]

test_file = './creditcard_test.csv'
test_ds = PeopleDataset(test_file)
test_loader = T.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)


def client_update(client_model, optimizer, train_loader, epoch=5):
    """
    This function updates/trains client model on client data
    """
    model.train()
    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = client_model(data)
            binary_loss = T.nn.BCEWithLogitsLoss()
            target = target.unsqueeze(1)
            target = target.float()
            loss = binary_loss(output, target)
            loss.backward()
            optimizer.step()
    return loss.item()


def server_aggregate_q(global_model, client_models, client_nums):
    """
    This function has aggregation method 'aggregated_quantize_encrypt'.
    """
    locals_w = [model.state_dict() for model in client_models]
    # aggregated_weights = aggregated_quantize_encrypt(locals_w, client_nums, args)
    aggregated_weights = aggregated_encrypt(locals_w, client_nums, args)
    global_model.load_state_dict(aggregated_weights)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())


def server_aggregate(global_model, client_models):
    """
    This function has aggregation method 'mean'
    """
    ### This will take simple mean of the weights of models ###
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = T.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(
            0)
    global_model.load_state_dict(global_dict)

    for model in client_models:
        model.load_state_dict(global_model.state_dict())


def test(global_model, test_loader):
    """This function tests the global model on test data and returns test loss and test accuracy """
    model.eval()
    test_loss = 0
    correct = 0
    actuals, predictions = list(), list()
    with T.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = global_model(data)
            binary_loss = T.nn.BCEWithLogitsLoss()
            target = target.unsqueeze(1)
            target = target.float()
            test_loss += binary_loss(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            actual = target.cpu().numpy()
            pr = output.detach().cpu().numpy()
            pr = pr.round()
            predictions.append(pr)
            actuals.append(actual)

    test_loss /= len(test_loader.dataset)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    # calculate precision
    prescision = precision_score(actuals, predictions)
    # calculate recall
    recall = recall_score(actuals, predictions)
    # calculate f1
    f1 = f1_score(actuals, predictions)
    fed_acc.append(acc)
    fed_pre.append(prescision)
    fed_recall.append(recall)
    fed_f1.append(f1)
    print()
    print(confusion_matrix(actuals, predictions))
    return test_loss, acc, prescision, recall, f1, actuals, predictions


###########################################
#### Initializing models and optimizer  ####
############################################

#### global model ##########
# global_model =  MLPUpdated().cuda()
global_model = MLPUpdated().to(device)

############## client models ##############
# client_models = [ MLPUpdated().cuda() for _ in range(num_selected)]
client_models = [MLPUpdated().to(device) for _ in range(num_selected)]
for model in client_models:
    model.load_state_dict(global_model.state_dict())  ### initial synchronizing with global model

############### optimizers ################
opt = [optim.SGD(model.parameters(), lr=args.lr) for model in client_models]

print(len(fed_acc))
print(len(fed_pre))
print(len(fed_recall))
print(len(fed_f1))

###### List containing info about learning #########
losses_train = []
losses_test = []
acc_train = []
acc_test = []
all_actuals, all_predictions = [], []
# Runnining FL

import time

start_time = time.time()
for r in range(num_rounds):
    start_time_r = time.time()
    # select random clients
    client_idx = np.random.permutation(num_clients)[:num_selected]
    # client update
    loss = 0
    for i in tqdm(range(num_selected)):
        loss += client_update(client_models[i], opt[i], train_loader[client_idx[i]], epoch=epochs)

    losses_train.append(loss)
    # server aggregate
    # server_aggregate(global_model, client_models)
    server_aggregate_q(global_model, client_models, num_selected)

    test_loss, acc, precision, recall, f1, actuals, predictions = test(global_model, test_loader)
    all_actuals.extend(actuals)
    all_predictions.extend(predictions)
    losses_test.append(test_loss)
    acc_test.append(acc)
    end_time_r = time.time()
    print('%d-th round' % r)
    print(
        'average train loss %0.3g | test loss %0.3g | test acc: %0.3f | test prescision: %0.3f | test recall: %0.3f | test f1: %0.3f' % (
        loss / num_selected, test_loss, acc, precision, recall, f1))
    print(f"Round {r} Execution time: {end_time_r - start_time_r} seconds")

print("--- %s seconds ---" % (time.time() - start_time))
print(len(fed_acc))
print(len(fed_pre))
print(len(fed_recall))
print(len(fed_f1))

mp[str(num_selected)] = {'fed_acc': fed_acc, 'fed_pre': fed_pre, 'fed_recall': fed_recall, 'fed_f1': fed_f1}
print(mp)
print(len(mp))

# To write the results in json file
data = mp
a_file = open(str(num_selected) + "_results.json", "w")
json.dump(data, a_file)
a_file.close()

# To read the results in json file
a_file = open(str(cln) + "_results.json", "r")
a_dictionary = json.load(a_file)

mp = a_dictionary
print(a_dictionary.keys())
print(len(a_dictionary[str(num_selected)]['fed_acc']))

i = cln
a_file = open(str(i) + "_results.json", "r")
a_dictionary = json.load(a_file)

mp = a_dictionary

import matplotlib.pyplot as plt

x = [i for i in range(1, num_rounds + 1)]

print("Number of Rounds: ", num_rounds)

save_dir = "save"
os.makedirs(save_dir, exist_ok=True)
img_name = f"encrypt__{args.num_clients}__frac{args.num_select}__{args.he_scheme}__{args.poly_modulus_degree}__round{args.num_rounds}_dataset_{args.dataset}"
fig = plt.figure()
fig.set_size_inches(25.5, 10.5)

plt.subplot(2, 2, 1)
plt.plot(x, mp[str(num_selected)]['fed_acc'])
plt.legend([str(num_selected)])
plt.ylabel("Accuracy")
plt.xlabel("Rounds")

plt.subplot(2, 2, 2)
plt.plot(x, mp[str(num_selected)]['fed_pre'])
plt.legend([str(num_selected)])
plt.ylabel("Precision")
plt.xlabel("Rounds")

plt.subplot(2, 2, 3)
plt.plot(x, mp[str(num_selected)]['fed_recall'])
plt.legend([str(num_selected)])
plt.ylabel("Recall")
plt.xlabel("Rounds")

plt.subplot(2, 2, 4)
plt.plot(x, mp[str(num_selected)]['fed_f1'])
plt.legend([str(num_selected)])
plt.ylabel("F1 score")
plt.xlabel("Rounds")
# plt.show()
plt.savefig(os.path.join(save_dir, img_name + '.jpg'))
plot_roc_curve(all_actuals, all_predictions)