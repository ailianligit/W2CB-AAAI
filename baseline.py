import os
import pandas as pd
import csv
import numpy as np

import torch
from torch.utils.data import TensorDataset

from utils import select_data_policy_based, get_ot_info, get_model_info, get_utility_info

from itertools import combinations, permutations
def LAVA_selection(sources_x, sources_y, test_loader, config):
    ot_dic = {}
    for combo in combinations(config['avalible'], config['round_budget']):
        print(combo)
        ot = get_ot_info(config, sources_x, sources_y, test_loader, combo)
        ot_dic[combo] = ot
    indices = min(ot_dic, key=ot_dic.get)
    policy = tuple(sorted(indices))
    return policy

import torchvision.models as models
import torch.nn as nn
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
def svm_model_log_err(device, train_loader, test_loader, label_num):
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

    train_data_embedding = torch.empty((0, 512))
    train_label_embedding = torch.empty((0, ))
    train_data_embedding = train_data_embedding.to(device)
    train_label_embedding = train_label_embedding.to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = embedder(inputs)
            train_data_embedding = torch.cat((train_data_embedding, outputs), dim=0)
            train_label_embedding = torch.cat((train_label_embedding, targets), dim=0)

    test_data_embedding = torch.empty((0, 512))
    test_label_embedding = torch.empty((0, ))
    test_data_embedding = test_data_embedding.to(device)
    test_label_embedding = test_label_embedding.to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = embedder(inputs)
            test_data_embedding = torch.cat((test_data_embedding, outputs), dim=0)
            test_label_embedding = torch.cat((test_label_embedding, targets), dim=0)

    x_train = train_data_embedding.cpu().numpy()
    y_train = train_label_embedding.cpu().numpy()
    x_test = test_data_embedding.cpu().numpy()
    y_test = test_label_embedding.cpu().numpy()

    svm = LinearSVC()
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

def compute_shapley(acc_dic, block_num):
    def v(S):
        v = acc_dic[tuple(sorted(S))]
        return v

    factorial = lambda x: 1 if x == 0 else x * factorial(x-1)
    fact = factorial(block_num)
    long_term_values = {i: 0 for i in range(block_num)}
    
    for r in range(1, block_num+1):
        for perm in permutations(range(block_num), r):
            for j, player in enumerate(perm):
                S = perm[:j]
                S_with_player = perm[:j+1]
                marginal_contribution = v(S_with_player) - v(S)
                long_term_values[player] += marginal_contribution / fact

    data_values = []
    for i in range(block_num):
        data_values.append(long_term_values[i])
    return data_values

def shapley_selection(sources_x, sources_y, test_loader, config):
    acc_dic = {}
    acc_dic[()] = 0.0
    for r in range(1, config['block_num']+1):
        for combo in combinations(range(config['block_num']), r):
            select_x, select_y = select_data_policy_based(sources_x, sources_y, combo)
            train_loader = torch.utils.data.DataLoader(dataset=TensorDataset(torch.Tensor(select_x), 
                                        torch.LongTensor(select_y)), 
                                        batch_size=config['batch_size'], 
                                        shuffle=True)
            acc = svm_model_log_err(config['device'], train_loader, test_loader, config['label_num'])
            acc_dic[combo] = acc
    data_values = compute_shapley(acc_dic, config['block_num'])
    shapley_path = f"/home/ubuntu/data/sim_selection/sim_shapley/{config['network']}_{config['dataset']}/size_{config['block_size']}_block_{config['block_num']}_domain_{config['domain_num']}.csv"
    with open(shapley_path, mode='a', newline='') as shapley_file:
        shapley_writer = csv.writer(shapley_file)
        shapley_writer.writerow(data_values)
    data_values_np = np.array(data_values)
    selected_arms = np.argsort(data_values_np)[-config['round_budget']:]
    policy = tuple(sorted(selected_arms))
    return policy

from sklearn.preprocessing import OneHotEncoder
def get_coefficient(device, loader, label_num):
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

    data_embedding = torch.empty((0, 512))
    data_embedding = data_embedding.to(device)
    label_embedding = torch.empty((0, ))
    label_embedding = label_embedding.to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = embedder(inputs)
            data_embedding = torch.cat((data_embedding, outputs), dim=0)
            label_embedding = torch.cat((label_embedding, targets), dim=0)

    all_data = data_embedding.cpu().numpy()
    all_labels = label_embedding.cpu().numpy()
    
    # 将数据展平
    num_samples = all_data.shape[0]
    num_features = np.prod(all_data.shape[1:])
    all_data_flat = all_data.reshape(num_samples, num_features)
    
    # 独热编码标签
    encoder = OneHotEncoder(sparse=False)
    all_labels_onehot = encoder.fit_transform(all_labels.reshape(-1, 1))
    
    # 计算相关系数矩阵
    correlation_matrix = []
    
    for i in range(all_labels_onehot.shape[1]):
        label_column = all_labels_onehot[:, i]
        correlation = np.corrcoef(all_data_flat.T, label_column)
        correlation_matrix.append(correlation[-1, :-1])  # 排除最后一个自身相关性
    
    correlation_matrix = np.array(correlation_matrix)
    
    # 处理 NaN 值
    correlation_matrix = np.nan_to_num(correlation_matrix)
    
    return correlation_matrix

from numpy.linalg import norm as l2_norm
def get_l2_similarity(coefficient1, coefficient2):
    # 计算L2范数
    l2_distance = l2_norm(coefficient1 - coefficient2)  
    return l2_distance

def get_cosine_similarity(coefficient1, coefficient2):
    # 计算余弦距离
    dot_product = np.sum(coefficient1 * coefficient2)
    norm1 = l2_norm(coefficient1)
    norm2 = l2_norm(coefficient2)
    cosine_similarity = dot_product / (norm1 * norm2)
    cosine_distance = 1 - cosine_similarity
    
    return cosine_distance

def correlation_selection(sources_x, sources_y, test_loader, config):
    train_coefficients = []
    for i in range(config['block_num']):
        train_loader = torch.utils.data.DataLoader(dataset=TensorDataset(torch.Tensor(sources_x[i]), 
                                    torch.LongTensor(sources_y[i])), 
                                    batch_size=config['batch_size'], 
                                    shuffle=True)
        coefficient = get_coefficient(config['device'], train_loader, config['label_num'])
        train_coefficients.append(coefficient)
                # correlation_writer.writerow(coefficient.tolist())

    coefficient = get_coefficient(config['device'], test_loader, config['label_num'])
    simi_list = []
    for i in range(config['block_num']):
        if config['method'] == 'norm':
            norm_l2 = get_l2_similarity(train_coefficients[i], coefficient)
            simi_list.append(norm_l2)
        elif config['method'] == 'cos':
            cos = get_cosine_similarity(train_coefficients[i], coefficient)
            simi_list.append(cos)
    correlation_path = f"/home/ubuntu/data/sim_selection/sim_{config['method']}/{config['network']}_{config['dataset']}/size_{config['block_size']}_block_{config['block_num']}_domain_{config['domain_num']}.csv"
    with open(correlation_path, mode='a', newline='') as correlation_file:
        correlation_writer = csv.writer(correlation_file)
        correlation_writer.writerow(simi_list)
    simi_list_np = np.array(simi_list)
    selected_arms = np.argsort(simi_list_np)[:config['round_budget']]
    policy = tuple(sorted(selected_arms))
    return policy

import random
from sklearn.linear_model import LinearRegression
def datamodels_selection(sources_x, sources_y, test_loader, config, iters=20):
    policy_lists = []
    loss_lists = []
    for _ in range(iters):
        original_tuple = tuple(range(config['block_num']))
        subset = random.sample(original_tuple, random.randint(1, len(original_tuple)))
        policy = tuple(sorted(subset))

        policy_list = [1 if j in policy else 0 for j in range(config['block_num'])]
        policy_lists.append(policy_list)

        model = get_model_info(config, sources_x, sources_y, policy)
        
        model.eval()
        test_criterion = nn.CrossEntropyLoss()
        loss_list = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(config['device']), targets.to(config['device'])
                outputs = model(inputs)
                loss = test_criterion(outputs, targets)
                loss_list.append(loss.item())

        loss_lists.append(loss_list)

    policy_array = np.array(policy_lists)
    coefficient_list = []
    for i in range(len(loss_lists[0])):
        column_i = [row[i] for row in loss_lists]
        loss_array = np.array(column_i).reshape(-1, 1)
        model = LinearRegression()
        model.fit(policy_array, loss_array)
        coefficients = model.coef_
        coefficient_list.append(coefficients.flatten().tolist())

    column_sums = [sum(row[i] for row in coefficient_list) for i in range(config['block_num'])]
    datamodels_path = f"/home/ubuntu/data/sim_selection/sim_datamodels/{config['network']}_{config['dataset']}/size_{config['block_size']}_block_{config['block_num']}_domain_{config['domain_num']}.csv"
    with open(datamodels_path, mode='a', newline='') as datamodels_file:
        datamodels_writer = csv.writer(datamodels_file)
        datamodels_writer.writerow(column_sums)
    min_sum_column_indices = np.argsort(column_sums)[:config['round_budget']]
    min_sum_column_indices_array = np.array(min_sum_column_indices)
    policy = tuple(sorted(min_sum_column_indices_array))
    return policy

def all_selection(sources_x, sources_y, test_loader, config):
    acc_dic = {}
    for combo in combinations(config['avalible'], config['round_budget']):
        acc, loss = get_utility_info(config, sources_x, sources_y, test_loader, combo)
        acc_dic[combo] = acc
    indices = max(acc_dic, key=acc_dic.get)
    policy = tuple(sorted(indices))
    return policy