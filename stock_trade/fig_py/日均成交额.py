import pandas as pd
import os

# 读取行业分类文件
try:
    industry_df = pd.read_csv('zz500行业分类.csv')
except FileNotFoundError:
    print("错误：找不到行业分类文件，请检查文件路径")
    exit()

# 提取证券代码中的数字部分并转换为数值类型
industry_df['证券代码'] = industry_df['证券代码'].str.extract('(\d+)', expand=False)
industry_df['证券代码'] = pd.to_numeric(industry_df['证券代码'], errors='coerce')


# 定义函数来处理单个股票文件
def process_stock_file(file_path):
    stock_code = os.path.basename(file_path).split('.')[0]

    try:
        df = pd.read_csv(file_path)

        # 计算日均成交额
        average_amount = df['成交额.元.'].mean()

        # 计算成交量波动率 (变异系数 = 标准差/均值)
        volume_std = df['成交量.股.'].std()
        volume_mean = df['成交量.股.'].mean()
        volume_cv = volume_std / volume_mean if volume_mean != 0 else float('inf')

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return stock_code, None, None

    return stock_code, average_amount, volume_cv


# 存储结果的列表
results = []

# 股票数据目录
stock_data_dir = '../raw_data_zz500成分股日数据'

# 检查目录是否存在
if not os.path.exists(stock_data_dir):
    print(f"错误：股票数据目录 {stock_data_dir} 不存在")
    exit()

# 遍历股票数据文件
for file in os.listdir(stock_data_dir):
    if file.endswith('.csv') and file != 'zz500行业分类.csv':
        file_path = os.path.join(stock_data_dir, file)
        stock_code, average_amount, volume_cv = process_stock_file(file_path)
        if average_amount is not None and volume_cv is not None:
            results.append([stock_code, average_amount, volume_cv])

# 将结果转换为 DataFrame
results_df = pd.DataFrame(results, columns=['股票代码', '日均成交额', '成交量波动率'])

# 转换股票代码数据类型
results_df['股票代码'] = pd.to_numeric(results_df['股票代码'])

# 合并行业分类信息
merged_df = pd.merge(results_df, industry_df, left_on='股票代码', right_on='证券代码', how='inner')

# 计算每个行业的平均波动率
industry_volatility = merged_df.groupby('所属行业')['成交量波动率'].mean().reset_index()
industry_volatility.columns = ['所属行业', '行业平均波动率']

# 将行业平均波动率合并到主DataFrame
merged_df = pd.merge(merged_df, industry_volatility, on='所属行业')

# 筛选同时满足两个条件的股票
# 条件1: 日均成交额 > 1亿元
# 条件2: 成交量波动率 < 行业平均波动率
filtered_df = merged_df[
    (merged_df['日均成交额'] > 100000000) &
    (merged_df['成交量波动率'] < merged_df['行业平均波动率'])
    ]

# 选择需要的列并保存结果
if not filtered_df.empty:
    output_df = filtered_df[['股票代码', '所属行业', '日均成交额', '成交量波动率', '行业平均波动率']]
    output_df = output_df.rename(columns={
        '所属行业': '股票所属行业',
        '成交量波动率': '个股波动率',
        '行业平均波动率': '所属行业平均波动率'
    })
    output_df.to_csv('筛选结果.csv', index=False)
    print(f"成功保存结果到 {os.path.abspath('筛选结果.csv')}")
    print(f"共筛选出 {len(output_df)} 只股票")

    # 按行业汇总统计信息（增加行业平均波动率）
    industry_summary = filtered_df.groupby('所属行业').agg({
        '股票代码': 'count',
        '日均成交额': 'mean',
        '成交量波动率': 'mean'  # 计算行业内符合条件股票的平均波动率
    }).reset_index()

    # 合并原始行业平均波动率（包含所有股票）
    industry_summary = pd.merge(
        industry_summary,
        industry_volatility.rename(columns={'行业平均波动率': '行业总体平均波动率'}),
        on='所属行业'
    )

    # 重命名列并转换单位
    industry_summary.columns = [
        '行业',
        '股票数量',
        '行业平均日均成交额(元)',
        '筛选后行业平均波动率',
        '行业总体平均波动率'
    ]

    # 将行业平均日均成交额转换为万元，并保留四位小数
    industry_summary['行业平均日均成交额(万元)'] = (industry_summary['行业平均日均成交额(元)'] / 10000).round(4)

    # 保留四位小数
    industry_summary['筛选后行业平均波动率'] = industry_summary['筛选后行业平均波动率'].round(4)
    industry_summary['行业总体平均波动率'] = industry_summary['行业总体平均波动率'].round(4)

    # 选择需要的列并排序
    final_summary = industry_summary[
        ['行业', '股票数量', '行业平均日均成交额(万元)', '筛选后行业平均波动率', '行业总体平均波动率']]

    # 保存行业汇总结果
    final_summary.to_csv('行业汇总结果.csv', index=False)
    print(f"行业汇总结果已保存到 {os.path.abspath('行业汇总结果.csv')}")
else:
    print("没有找到符合条件的股票")