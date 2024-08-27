import argparse
import random
import csv
import numpy as np
import os
import time
import joblib

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split, TensorDataset

from utils import get_utility_info, get_embedding_info, get_task_labels, get_ot_sources, get_source_labels
from baseline import LAVA_selection, shapley_selection, correlation_selection, datamodels_selection, all_selection

batch_size = 64
test_block_size = 200
# CIFAR-10 dataset
cifar10_data_dir = '/home/ubuntu/data/dataset'
# PACS dataset
domains = ['cartoon', 'art_painting', 'sketch', 'photo']
pacs_data_dir = '/home/ubuntu/data/dataset/PACS'

domain_block_num = [[9,0,0,0],[5,4,0,0],[3,3,3,0],[3,2,2,2]]
all_block=[[0,1,2],[3,4,5],[6,7,8],[9,10,11]]

def dataset_q(label_num, q1_amt, q2_amt, num, train_feats, label_idx):
    ds1_idx = []
    ds2_idx = []
    ds3_idx = []
    ds1_labels = []
    ds2_labels = []
    ds3_labels = []

    if label_num == 7:
        d1c1 = 0.475
        d1c2 = 0.01
        d1c3 = 0.01

        d2c1 = 0.01
        d2c2 = 0.475
        d2c3 = 0.01

        d3c1 = 0.01
        d3c2 = 0.01
        d3c3 = 0.32
    elif label_num == 10:
        d1c1 = 0.32
        d1c2 = 0.0057
        d1c3 = 0.0057

        d2c1 = 0.0057
        d2c2 = 0.32
        d2c3 = 0.0057

        d3c1 = 0.005
        d3c2 = 0.005
        d3c3 = 0.2425
    
    # sample size
    n = num # size of dataset for training (use for construct)
    # ratio
    q1 = q1_amt # q * dataset 1
    q2 = q2_amt # q * dataset 1
    q3 = 1-q1-q2 # q * dataset 1

    for i in range(int(label_num/3)):
        ds1_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q1*d1c1)))])
        ds2_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q2*d2c1)))])
        ds3_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q3*d3c1)))])
        ds1_labels.append(np.ones(int(np.rint(n*q1*d1c1)))*i)
        ds2_labels.append(np.ones(int(np.rint(n*q2*d2c1)))*i)
        ds3_labels.append(np.ones(int(np.rint(n*q3*d3c1)))*i)
    for i in range(int(label_num/3), int(2*label_num/3)):
        ds1_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q1*d1c2)))])
        ds2_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q2*d2c2)))])
        ds3_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q3*d3c2)))])
        ds1_labels.append(np.ones(int(np.rint(n*q1*d1c2)))*i)
        ds2_labels.append(np.ones(int(np.rint(n*q2*d2c2)))*i)
        ds3_labels.append(np.ones(int(np.rint(n*q3*d3c2)))*i)
    for i in range(int(2*label_num/3), label_num):
        ds1_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q1*d1c3)))])
        ds2_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q2*d2c3)))])
        ds3_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q3*d3c3)))])
        ds1_labels.append(np.ones(int(np.rint(n*q1*d1c3)))*i)
        ds2_labels.append(np.ones(int(np.rint(n*q2*d2c3)))*i)
        ds3_labels.append(np.ones(int(np.rint(n*q3*d3c3)))*i)

    ds1_features_fl = train_feats[np.concatenate(ds1_idx)]
    ds2_features_fl = train_feats[np.concatenate(ds2_idx)]
    ds3_features_fl = train_feats[np.concatenate(ds3_idx)]

    ds1_labels = np.concatenate(ds1_labels)
    ds2_labels = np.concatenate(ds2_labels)
    ds3_labels = np.concatenate(ds3_labels)

    train_x = np.concatenate([ds1_features_fl, ds2_features_fl, ds3_features_fl])
    train_y = np.concatenate([ds1_labels, ds2_labels, ds3_labels])
            
    return train_x, train_y
 
def split_train_task_data_cifar10():
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_data = datasets.CIFAR10(
        root=cifar10_data_dir, download=True, train=False, transform=transform)
    test_data = datasets.CIFAR10(
        root=cifar10_data_dir, download=True, train=True, transform=transform)
    task_data_list = [test_data]
    return train_data, task_data_list

def split_train_task_data_pacs(domain_num, train_ratio=0.5):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_data_list = []
    task_data_list = []
    for i in range(domain_num):
        domain_path = os.path.join(pacs_data_dir, domains[i])
        dataset = datasets.ImageFolder(root=domain_path, transform=transform)
        train_size = int(train_ratio * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_data_list.append(train_dataset)
        task_data_list.append(test_dataset)
    return train_data_list, task_data_list

def generate_valid_tuple():
    numbers = [float(f"{i * 0.1:.1f}") for i in range(11)]
    while True:
        num1 = random.choice(numbers)
        num2 = random.choice(numbers)
        if num1 + num2 <= 1:
            return (num1, num2)

def split_train_data_cifar10(file, train_data, block_size, block_num, label_num):
    file.write('\nsplit_train_data:\n')
    
    features = []
    labels = []
    for img, label in train_data:
        features.append(img.numpy())
        labels.append(label)
    train_features = np.array(features)
    train_labels = np.array(labels)

    sources_x = []
    sources_y = []
    label_idx = []
    for i in range(10):
        label_idx.append((train_labels==i).nonzero()[0])
    for k in range(block_num):
        p = generate_valid_tuple()
        train_x, train_y = dataset_q(label_num, p[0], p[1], block_size, train_features, label_idx)
        unique_elements = np.unique(train_y)
        file.write(f'No.{k+1}: p1={p[0]}, p2={p[1]}, {unique_elements.size} labels\n')
        sources_x.append(train_x)
        sources_y.append(train_y)
    file.flush()
    return sources_x, sources_y

def split_train_data_pacs(file, train_data_list, block_size, block_num, domain_num, label_num):
    file.write('\nsplit_train_data:\n')
    sources_x = []
    sources_y = []
    for i in range(domain_num):
        # 初始化空列表来存储特征和标签
        features = []
        labels = []
        # 遍历数据集提取特征和标签
        for img, label in train_data_list[i]:
            features.append(img.numpy())  # 将图像张量转换为 NumPy 数组
            labels.append(label)
        # 将列表转换为 NumPy 数组
        features = np.array(features)
        labels = np.array(labels)
        label_idx = []
        for j in range(label_num):
            label_idx.append((labels==j).nonzero()[0])
        for k in range(domain_block_num[domain_num-1][i]):
            p = generate_valid_tuple()
            train_x, train_y = dataset_q(label_num, p[0], p[1], block_size, features, label_idx)
            unique_elements = np.unique(train_y)
            file.write(f'{domains[i]}: No.{k+1}, p1={p[0]}, p2={p[1]}, {unique_elements.size} labels\n')
            sources_x.append(train_x)
            sources_y.append(train_y)
    file.flush()
    return sources_x, sources_y

def task_sequence(file, test_duration, max_budegt, domain_num):
    file.write('\ntask_sequence:\n')
    budget_list = [random.randint(1, max_budegt) for _ in range(test_duration)]
    domain_list = [random.randint(0, domain_num-1) for _ in range(test_duration)]
    q_list = [generate_valid_tuple() for _ in range(test_duration)]
    file.write(f'budget_list: {budget_list}\n')
    file.write(f'domain_list: {domain_list}\n')
    file.write(f'q_list: {q_list}\n')
    file.flush()
    return budget_list, domain_list, q_list

def get_avalible_cifar10(budget_list, duration, block_num):
    file.write('\nget_avalible:\n')
    avalible_list = []
    for i in range(duration):
        budget = budget_list[i]
        new_list = random.sample(range(block_num), k=random.randint(budget, block_num))
        available_array=tuple(sorted(new_list))
        avalible_list.append(available_array)
    file.write(f'available_array: {avalible_list}\n')
    file.flush()
    return avalible_list

def get_avalible_pacs(budget_list, domain_list, duration, domain_num):
    file.write('\nget_avalible:\n')
    avalible_list = []
    for i in range(duration):
        budget = budget_list[i]
        domain = domain_list[i]
        available = []
        for j in range(domain_num):
            if j == domain:
                new_list = random.sample(all_block[j], k=random.randint(budget, domain_block_num[domain_num-1][j]))
                available.extend(new_list)
            else:
                new_list = random.sample(all_block[j], k=random.randint(1, domain_block_num[domain_num-1][j]))
                available.extend(new_list)
            available_array=tuple(sorted(available))
        avalible_list.append(available_array)
    file.write(f'available_array: {avalible_list}\n')
    file.flush()
    return avalible_list

def generate_policy(sources_x, sources_y, test_loader, config):
    if config['method'] == 'random':
        sample = random.sample(range(config['block_num']), config['round_budget'])
        policy = tuple(sorted(sample))
    elif config['method'] == 'LAVA':
        policy = LAVA_selection(sources_x, sources_y, test_loader, config)
    elif config['method'] == 'shapley':
        policy = shapley_selection(sources_x, sources_y, test_loader, config)
    elif config['method'] == 'norm' or config['method'] == 'cos':
        policy = correlation_selection(sources_x, sources_y, test_loader, config)
    elif config['method'] == 'datamodels':
        policy = datamodels_selection(sources_x, sources_y, test_loader, config)
    elif config['method'] == 'all':
        policy = all_selection(sources_x, sources_y, test_loader, config)
    return policy

if __name__ == "__main__":
    print("Available CUDA devices: ", torch.cuda.device_count())
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available. PyTorch will use the CPU.")

    def validate_input(value):
        if value == "random":
            return value
        try:
            int_value = int(value)
            return int_value
        except ValueError:
            raise argparse.ArgumentTypeError("Input must be 'random' or an integer.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method", choices=["labels", "random", "LAVA", "shapley", "norm", "cos", "datamodels", "all", "embedding"], required=True)
    parser.add_argument("--budget", type=validate_input, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--block-num", type=int, required=True)
    parser.add_argument("--domain-num", type=int, required=True)
    parser.add_argument("--train-begin", type=int, required=True)
    parser.add_argument("--test-begin", type=int, required=True)
    parser.add_argument("--train-duration", type=int, required=True)
    parser.add_argument("--test-duration", type=int, required=True)
    parser.add_argument("--network", choices=["mobilenetv2", "resnet18", "densenet121"], required=True)
    parser.add_argument("--dataset", choices=["cifar10", "pacs"], required=True)
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    method = args.method
    budget = args.budget
    block_size = args.block_size
    block_num = args.block_num
    domain_num = args.domain_num
    train_begin = args.train_begin
    test_begin = args.test_begin
    train_duration = args.train_duration
    test_duration = args.test_duration
    network = args.network
    dataset = args.dataset

    if dataset == 'pacs':
        label_num = 7
        max_budegt = 3
    elif dataset == 'cifar10':
        label_num = 10
        domain_num = 1
        max_budegt = 6
    train_results_path = f'/home/ubuntu/data/sim_selection/train_results/{network}_{dataset}/dynamic_{method}_budget_{budget}_size_{block_size}_block_{block_num}_domain_{domain_num}.csv'
    test_results_path = f'/home/ubuntu/data/sim_selection/test_results/{network}_{dataset}/dynamic_{method}_budget_{budget}_size_{block_size}_block_{block_num}_domain_{domain_num}.csv'    

    config = {}
    config['method'] = method
    config['budegt'] = budget
    config['block_size'] = block_size
    config['block_num'] = block_num
    config['batch_size'] = batch_size
    config['network'] = network
    config['dataset'] = dataset
    config['device'] = device
    config['domain_num'] = domain_num
    config['label_num'] = label_num
    config['max_budegt'] = max_budegt
    config['train_results_path'] = train_results_path
    config['test_results_path'] = test_results_path
    
    with open(f'/home/ubuntu/data/sim_selection/log_cache/dynamic_{network}_{dataset}_{method}_budget_{budget}_size_{block_size}_block_{block_num}_domain_{domain_num}_{int(time.time())}.txt', 'w', encoding='utf-8') as file:
        file.write(f'selection config: \n')
        file.write(f'seed: {seed}\n')
        file.write(f'method: {method}\n')
        file.write(f'budget: {budget}\n')
        file.write(f'block_size: {block_size}\n')
        file.write(f'block_num: {block_num}\n')
        file.write(f'train_begin: {train_begin}\n')
        file.write(f'test_begin: {test_begin}\n')
        file.write(f'train_duration: {train_duration}\n')
        file.write(f'test_duration: {test_duration}\n')
        file.write(f'batch_size: {batch_size}\n')
        file.write(f'network: {network}\n')
        file.write(f'dataset: {dataset}\n')
        file.write(f'domain_num: {domain_num}\n')
        file.write(f'label_num: {domain_num}\n')
        file.write(f'max_budegt: {max_budegt}\n')
        file.write(f'train_results_path: {train_results_path}\n')
        file.write(f'test_results_path: {test_results_path}\n')
        file.flush()
        
        if dataset == 'pacs':
            train_data_list, task_data_list = split_train_task_data_pacs(domain_num)
            sources_x, sources_y = split_train_data_pacs(file, train_data_list, block_size, block_num, domain_num, label_num)
            test_budget_list, test_domain_list, test_q_list = task_sequence(file, test_duration, max_budegt, domain_num)
            train_budget_list, train_domain_list, train_q_list = task_sequence(file, train_duration, max_budegt, domain_num)
            test_avalible_list = get_avalible_pacs(test_budget_list, test_domain_list, test_duration, domain_num)
            train_avalible_list = get_avalible_pacs(train_budget_list, train_domain_list, train_duration, domain_num)
        elif dataset == 'cifar10':
            train_data, task_data_list = split_train_task_data_cifar10()
            sources_x, sources_y = split_train_data_cifar10(file, train_data, block_size, block_num, label_num)
            test_budget_list, test_domain_list, test_q_list = task_sequence(file, test_duration, max_budegt, domain_num)
            train_budget_list, train_domain_list, train_q_list = task_sequence(file, train_duration, max_budegt, domain_num)
            test_avalible_list = get_avalible_cifar10(test_budget_list, test_duration, block_num)
            train_avalible_list = get_avalible_cifar10(train_budget_list, train_duration, block_num)
        
    if method == 'all' or method == 'random' or method == 'labels' or method == 'embedding':
        # if method == 'labels':
        #     get_source_labels(sources_y, config)
        # elif method == 'embedding':
        #     get_ot_sources(sources_x, sources_y, config)

        with open(train_results_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            train_begin = args.train_begin
            for i in range(train_begin, train_duration):
                if budget == "random":
                    round_budget = train_budget_list[i]
                else:
                    round_budget = budget
                q1, q2 = train_q_list[i]
                domain = train_domain_list[i]
                avalible = train_avalible_list[i]
                config['round'] = i
                config['round_budget'] = round_budget
                config['q1'] = q1
                config['q2'] = q2
                config['domain'] = domain
                config['avalible'] = avalible
                print(f"duration: {i}, budget: {round_budget}, q1: {q1}, q2: {q2}, domain: {domain}, avalible: {avalible}")

                features = []
                labels = []
                for img, label in task_data_list[domain]:
                    features.append(img.numpy())
                    labels.append(label)
                features = np.array(features)
                labels = np.array(labels)
                label_idx = []
                for j in range(label_num):
                    label_idx.append((labels==j).nonzero()[0])
                test_x, test_y = dataset_q(label_num, q1, q2, test_block_size, features, label_idx)

                if method == 'labels':
                    get_task_labels(test_y, config)
                    continue

                test_loader = torch.utils.data.DataLoader(dataset=TensorDataset(torch.Tensor(test_x), 
                                                torch.LongTensor(test_y)), 
                                                batch_size=batch_size, 
                                                shuffle=False)
                
                if method == 'embedding':
                    get_embedding_info(sources_x, sources_y, test_loader, config)
                    continue
                
                policy = generate_policy(sources_x, sources_y, test_loader, config)
                acc, loss = get_utility_info(config, sources_x, sources_y, test_loader, policy)

                print(f"policy: {policy}, acc: {acc}, loss: {loss}")
                writer.writerow([i, round_budget, q1, q2, domain, f'{policy}', acc, loss])
                file.flush()

    with open(test_results_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        test_begin = args.test_begin
        for i in range(test_begin, test_duration):
            if budget == "random":
                round_budget = test_budget_list[i]
            else:
                round_budget = budget
            q1, q2 = test_q_list[i]
            domain = test_domain_list[i]
            avalible = test_avalible_list[i]
            config['round'] = i
            config['round_budget'] = round_budget
            config['q1'] = q1
            config['q2'] = q2
            config['domain'] = domain
            config['avalible'] = avalible
            print(f"duration: {i}, budget: {round_budget}, q1: {q1}, q2: {q2}, domain: {domain}, avalible: {avalible}")

            features = []
            labels = []
            for img, label in task_data_list[domain]:
                features.append(img.numpy())
                labels.append(label)
            features = np.array(features)
            labels = np.array(labels)
            label_idx = []
            for j in range(label_num):
                label_idx.append((labels==j).nonzero()[0])
            test_x, test_y = dataset_q(label_num, q1, q2, test_block_size, features, label_idx)

            if method == 'labels':
                get_task_labels(test_y, config)
                continue

            if method == 'datamodels':
                test_loader = torch.utils.data.DataLoader(dataset=TensorDataset(torch.Tensor(test_x), 
                                            torch.LongTensor(test_y)), 
                                            batch_size=1, 
                                            shuffle=False)
            else:
                test_loader = torch.utils.data.DataLoader(dataset=TensorDataset(torch.Tensor(test_x), 
                                            torch.LongTensor(test_y)), 
                                            batch_size=batch_size, 
                                            shuffle=False)
                
            if method == 'embedding':
                get_embedding_info(sources_x, sources_y, test_loader, config)
                continue
            
            policy = generate_policy(sources_x, sources_y, test_loader, config)
            acc, loss = get_utility_info(config, sources_x, sources_y, test_loader, policy)

            print(f"policy: {policy}, acc: {acc}, loss: {loss}")
            writer.writerow([i, round_budget, q1, q2, domain, f'{policy}', acc, loss])
            file.flush()