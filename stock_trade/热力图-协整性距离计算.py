import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import coint


def format_stock_code(code: str) -> str:
    # 去除首尾单引号（如果存在）
    code = code.strip("'")

    # 识别市场代码（前两位或后两位）
    if code[:2].isalpha():
        market = code[:2]
        number = code[2:]
    elif code[-2:].isalpha():
        market = code[-2:]
        number = code[:-2]
    else:
        raise ValueError("无法识别市场代码，格式应为'MKTXXXXXX'或'XXXXXXMKT'")

    # 返回格式化后的代码
    return f"{number}.{market}"


# 读取行业分类数据
industry_df = pd.read_csv('./zz500行业分类.csv')

# 获取指定目录下的所有文件
all_files = os.listdir('data_trading')

# 筛选出符合条件的文件路径（假设文件格式为 *.csv）
file_paths = [os.path.join('data_trading', file) for file in all_files if file.endswith('.csv')]

all_data = []
for file_path in file_paths:
    try:
        # 读取单个股票文件数据
        stock_df = pd.read_csv(file_path)

        # 提取股票代码，假设代码格式为 SZ000001 这种形式，你可能需要根据实际情况调整
        stock_code = stock_df['代码'].iloc[0]
        stock_code = format_stock_code(stock_code)

        # 查找对应的行业
        industry = industry_df[industry_df['证券代码'] == stock_code]['所属行业'].values[0]

        # 添加行业信息到股票数据中
        stock_df['所属行业'] = industry

        all_data.append(stock_df)
    except Exception as e:
        print(f'处理文件 {file_path} 时出现错误: {e}')

# 合并所有股票数据
merged_data = pd.concat(all_data, ignore_index=True)

# 确保数据按时间排序
merged_data['时间'] = pd.to_datetime(merged_data['时间'])
merged_data = merged_data.sort_values('时间')

# 创建行业-时间矩阵，缺失值用NaN填充
industry_pivot = merged_data.pivot_table(
    index='时间',
    columns='所属行业',
    values='收盘价'
)

# 获取行业名称列表
industries = industry_pivot.columns.tolist()
num_industries = len(industries)

# 初始化协整性检验结果矩阵
cointegration_matrix = np.zeros((num_industries, num_industries))

# 进行协整性检验并填充矩阵
for i in range(num_industries):
    for j in range(i, num_industries):
        if i == j:
            cointegration_matrix[i, j] = 0
        else:
            series1 = industry_pivot[industries[i]].ffill()  # 用前值填充
            series2 = industry_pivot[industries[j]].ffill()  # 用前值填充
            _, p_value, _ = coint(series1, series2)
            cointegration_matrix[i, j] = p_value
            cointegration_matrix[j, i] = p_value

# 将协整性检验结果矩阵转换为距离矩阵
# 这里简单地将p值作为距离度量，p值越大表示距离越远（协整性越弱）
distance_matrix = cointegration_matrix

# 将距离矩阵转换为DataFrame并保存为CSV文件
distance_df = pd.DataFrame(distance_matrix, index=industries, columns=industries)
distance_df.to_csv('./协整性距离.csv')