import numpy as np

import torch
import torchvision.models as models
import torch.nn as nn

from otdd.pytorch.distance import DatasetDistance, FeatureCost

def select_data_policy_based(sources_x, sources_y, policy):
    x = []
    y = []
    for i in policy:
        x.extend(sources_x[i])
        y.extend(sources_y[i])
    select_x = np.array(x)
    select_y = np.array(y)
    return select_x, select_y

def get_ot_dist(device, train_loader, test_loader, label_num, required_dual=False):
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, label_num)
    if label_num == 7:
        model.load_state_dict(torch.load('/home/ubuntu/data/sim_selection/checkpoint/pretrain_resnet18.pth', map_location=str('cuda')))
    else:
        model.load_state_dict(torch.load('/home/ubuntu/data/sim_selection/checkpoint/pretrain_resnet18_cifar10.pth', map_location=str('cuda')))
    model.eval()

    embedder = model.to(device)
    embedder.fc = torch.nn.Identity()
    for p in embedder.parameters():
        p.requires_grad = False

    # Here we use same embedder for both datasets
    if label_num == 7:
        feature_cost = FeatureCost(src_embedding = embedder,
                                src_dim = (3,224,224),
                                tgt_embedding = embedder,
                                tgt_dim = (3,224,224),
                                p = 2,
                                device=device)
    else:
        feature_cost = FeatureCost(src_embedding = embedder,
                                src_dim = (3,32,32),
                                tgt_embedding = embedder,
                                tgt_dim = (3,32,32),
                                p = 2,
                                device=device)

    dist = DatasetDistance(train_loader, test_loader,
                           inner_ot_method = 'exact',
                           debiased_loss = True,
                           feature_cost = feature_cost,
                           λ_x=1.0, λ_y=1.0,
                           sqrt_method = 'spectral',
                           sqrt_niters=10,
                           precision='single',
                           p = 2, entreg = 1e-2,
                           device=device)

    k = dist.distance(maxsamples=200, return_coupling=False)

    if required_dual == True:
        dual_sol = dist.dual_sol(maxsamples=1000, return_coupling=False)
        return k.item(), dual_sol

    # k = dist.distance(maxsamples=1000, return_coupling=False)
    return k.item()

import time
import torch.optim as optim
def train_model(device, train_loader, network, label_num, learning_rate=0.001, epoch=10):
    if network == 'resnet18':
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, label_num)                         
    elif network == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, label_num)
    elif network == 'mobilenetv2':
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, label_num)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for ep in range(epoch):
        start_time = time.time()
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            if inputs.shape[0] > 1:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        end_time = time.time()
        print('%.1f . TrainLoss: %.3f | TrainAcc: %.3f%% (%d/%d) | Time Elapsed %.3f sec ' % (ep, train_loss/(batch_idx+1), 100.*correct/total, correct, total, end_time-start_time))
    return model

def get_model_log_err(model, test_loader, device):
    model.eval()
    test_criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = test_criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print('TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return 100.*correct/total, test_loss/(batch_idx+1)

def get_model_info(config, sources_x, sources_y, policy):
    policy_list = [1 if j in policy else 0 for j in range(config['block_num'])]
    number_string = ''.join(str(num) for num in policy_list)
    model_name = f"size_{config['block_size']}_block_{config['block_num']}_domain_{config['domain_num']}_{number_string}.pth"
    model_path = f"/home/ubuntu/data/sim_selection/sim_model_cache/{config['network']}_{config['dataset']}/{model_name}"
    if os.path.isfile(model_path):
        if config['network'] == 'resnet18':
            model = models.resnet18()
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, config['label_num'])                         
        elif config['network'] == 'densenet121':
            model = models.densenet121()
            model.classifier = nn.Linear(model.classifier.in_features, config['label_num'])
        elif config['network'] == 'mobilenetv2':
            model = models.mobilenet_v2()
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, config['label_num'])
        print(model_name)
        model.load_state_dict(torch.load(model_path))
        model = model.to(config['device'])
    else:
        select_x, select_y = select_data_policy_based(sources_x, sources_y, policy)
        train_loader = torch.utils.data.DataLoader(dataset=TensorDataset(torch.Tensor(select_x), 
                                    torch.LongTensor(select_y)), 
                                    batch_size=config['batch_size'], 
                                    shuffle=True)
        model = train_model(config['device'], train_loader, config['network'], config['label_num'])
        torch.save(model.state_dict(), model_path)
    return model

import os
import csv
import pandas as pd
from torch.utils.data import TensorDataset
def get_utility_info(config, sources_x, sources_y, test_loader, policy):
    utility_path = f"/home/ubuntu/data/sim_selection/sim_utility_cache/{config['network']}_{config['dataset']}/size_{config['block_size']}_block_{config['block_num']}_domain_{config['domain_num']}.csv"
    utility_flag = False
    if os.path.isfile(utility_path):
        utility_df = pd.read_csv(utility_path, header=None)
        for index, row in utility_df.iterrows():
            if row[3] == str(policy) and row[0] == config['q1'] and row[1] == config['q2'] and row[2] == config['domain']:
                utility_flag = True
                break
    if utility_flag:
        acc = utility_df.iloc[index, 4]
        loss = utility_df.iloc[index, 5]
    else:
        model = get_model_info(config, sources_x, sources_y, policy)
        acc, loss = get_model_log_err(model, test_loader, config['device'])
        with open(utility_path, mode='a', newline='') as utility_file:
            utility_writer = csv.writer(utility_file)
            utility_writer.writerow([config['q1'], config['q2'], config['domain'], f'{policy}', acc, loss])

    return acc, loss

def get_ot_info(config, sources_x, sources_y, test_loader, policy):
    ot_path = f"/home/ubuntu/data/sim_selection/sim_ot_cache/{config['network']}_{config['dataset']}/size_{config['block_size']}_block_{config['block_num']}_domain_{config['domain_num']}.csv"
    ot_flag = False
    if os.path.isfile(ot_path):
        ot_df = pd.read_csv(ot_path, header=None)
        for index, row in ot_df.iterrows():
            if row[3] == str(policy) and row[0] == config['q1'] and row[1] == config['q2'] and row[2] == config['domain']:
                ot_flag = True
                break
    if ot_flag:
        ot = ot_df.iloc[index, 4]
    else:
        select_x, select_y = select_data_policy_based(sources_x, sources_y, policy)
        train_loader = torch.utils.data.DataLoader(dataset=TensorDataset(torch.Tensor(select_x), 
                                    torch.LongTensor(select_y)), 
                                    batch_size=config['batch_size'], 
                                    shuffle=True)
        ot = get_ot_dist(config['device'], train_loader, test_loader, config['label_num'])
        with open(ot_path, mode='a', newline='') as ot_file:
            ot_writer = csv.writer(ot_file)
            ot_writer.writerow([config['q1'], config['q2'], config['domain'], f'{policy}', ot])
    
    return ot

def get_embedding(device, train_loader, test_loader, label_num):
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, label_num)
    if label_num == 7:
        model.load_state_dict(torch.load('/home/ubuntu/data/sim_selection/checkpoint/pretrain_resnet18.pth', map_location=str('cuda')))
    else:
        model.load_state_dict(torch.load('/home/ubuntu/data/sim_selection/checkpoint/pretrain_resnet18_cifar10.pth', map_location=str('cuda')))
    model.eval()

    embedder = model.to(device)
    embedder.fc = torch.nn.Identity()
    for p in embedder.parameters():
        p.requires_grad = False

    # Here we use same embedder for both datasets
    if label_num == 7:
        feature_cost = FeatureCost(src_embedding = embedder,
                                src_dim = (3,224,224),
                                tgt_embedding = embedder,
                                tgt_dim = (3,224,224),
                                p = 2,
                                device=device)
    else:
        feature_cost = FeatureCost(src_embedding = embedder,
                                src_dim = (3,32,32),
                                tgt_embedding = embedder,
                                tgt_dim = (3,32,32),
                                p = 2,
                                device=device)

    dist = DatasetDistance(train_loader, test_loader,
                           inner_ot_method = 'exact',
                           debiased_loss = True,
                           feature_cost = feature_cost,
                           λ_x=1.0, λ_y=1.0,
                           sqrt_method = 'spectral',
                           sqrt_niters=10,
                           precision='single',
                           p = 2, entreg = 1e-2,
                           device=device)

    extracted_elements = dist.get_label_distance(maxsamples=200)

    return extracted_elements

def get_embedding_info(sources_x, sources_y, test_loader, config):
    embedding_path = f"/home/ubuntu/data/sim_selection/sim_embedding/{config['network']}_{config['dataset']}/size_{config['block_size']}_block_{config['block_num']}_domain_{config['domain_num']}_200.csv"
    
    embedding_lists = []
    for i in range(config['block_num']):
        train_loader = torch.utils.data.DataLoader(dataset=TensorDataset(torch.Tensor(sources_x[i]), 
                                    torch.LongTensor(sources_y[i])), 
                                    batch_size=config['batch_size'], 
                                    shuffle=True)
        extracted_elements = get_embedding(config['device'], train_loader, test_loader, config['label_num'])
        embedding_list = extracted_elements.tolist()
        embedding_lists.extend(embedding_list)

    with open(embedding_path, mode='a', newline='') as embedding_file:
        embedding_writer = csv.writer(embedding_file)
        embedding_writer.writerow(embedding_lists)

def get_task_labels(test_y, config):
    path = f"/home/ubuntu/data/sim_selection/sim_task_labels/{config['network']}_{config['dataset']}/size_{config['block_size']}_block_{config['block_num']}_domain_{config['domain_num']}.csv"
    
    counts = [0] * config['label_num']
    for number in range(config['label_num']):
        counts[number] = np.sum(test_y == number)

    with open(path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(counts)
        file.flush()

def get_source_labels(sources_y, config):
    path = f"/home/ubuntu/data/sim_selection/sim_source_labels/{config['network']}_{config['dataset']}/size_{config['block_size']}_block_{config['block_num']}_domain_{config['domain_num']}.csv"
    
    with open(path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for i in range(config['block_num']):
            counts = [0] * config['label_num']
            for number in range(config['label_num']):
                counts[number] = np.sum(sources_y[i] == number)
            writer.writerow(counts)
            file.flush()

def get_ot_sources(sources_x, sources_y, config):
    path = f"/home/ubuntu/data/sim_selection/sim_ot_sources/{config['network']}_{config['dataset']}/size_{config['block_size']}_block_{config['block_num']}_domain_{config['domain_num']}_200.csv"
    
    with open(path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for i in range(config['block_num']):
            for j in range(config['block_num']):
                unique_elements = np.unique(sources_y[i])
                print(f'{unique_elements.size} labels')
                unique_elements = np.unique(sources_y[j])
                print(f'{unique_elements.size} labels')
                train_loader = torch.utils.data.DataLoader(dataset=TensorDataset(torch.Tensor(sources_x[i]), 
                                        torch.LongTensor(sources_y[i])), 
                                        batch_size=config['batch_size'], 
                                        shuffle=True)
                test_loader = torch.utils.data.DataLoader(dataset=TensorDataset(torch.Tensor(sources_x[j]), 
                                        torch.LongTensor(sources_y[j])), 
                                        batch_size=config['batch_size'], 
                                        shuffle=True)
                ot = get_ot_dist(config['device'], train_loader, test_loader, config['label_num'])
                writer.writerow([i, j, ot])
                file.flush()