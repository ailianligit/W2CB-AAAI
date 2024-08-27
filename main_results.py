import pandas as pd

method = "ucb"   # "random", "LAVA", "shapley", "norm", "cos", "datamodels", "all", "embedding", "labels"
budget = "random"
size = 300
block = 9
domain = 3
network = "densenet121" # "resnet18", "densenet121", "mobilenetv2"
dataset = "pacs"    # "pacs", "cifar10"

def read_last_100_rows_and_calculate_mean(file_path, column_index):
    # 读取CSV文件
    df = pd.read_csv(file_path, header=None)
    
    # 获取最后100行数据
    last_100_rows = df.tail(100)
    
    # 获取指定列的数据
    column_data = last_100_rows.iloc[:, column_index]
    
    # 计算平均值
    mean_value = column_data.mean()
    
    return mean_value

file_path = f'/home/ubuntu/data/sim_selection/test_results/{network}_{dataset}/dynamic_{method}_budget_{budget}_size_{size}_block_{block}_domain_{domain}.csv'

# 示例用法

column_index = 6  # 第6列的索引是5（从0开始计数）
mean_value = read_last_100_rows_and_calculate_mean(file_path, column_index)
print(f"The mean value of the 6th column in the last 100 rows is: {mean_value}")