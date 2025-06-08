import os
import pandas as pd
from datetime import datetime
from pair_trading import PairTradingStrategy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import numpy as np
import argparse


# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='配对交易策略回测')
    parser.add_argument('--start_date', type=str, default='2024-01-01',
                        help='回测开始日期，格式: YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, default='2025-04-30',
                        help='回测结束日期，格式: YYYY-MM-DD')
    parser.add_argument('--output_dir', type=str, default='../fig/results',
                        help='结果输出目录')
    return parser.parse_args()


args = parse_args()

# 创建输出目录
os.makedirs(args.output_dir, exist_ok=True)

# 1. 加载行业分类数据
industry_df = pd.read_csv("zz500行业分类.csv", encoding='utf-8')
industry_df = industry_df.rename(columns={
    "证券代码": "stock_code",
    "证券简称": "stock_name",
    "上市日期": "list_date",
    "所属行业": "industry"
})

# 2. 加载股票价格数据
stock_folder = "../data_trading"
stock_data = {}

for filename in os.listdir(stock_folder):
    if filename.endswith(".csv"):
        stock_code = filename.replace(".csv", "")
        file_path = os.path.join(stock_folder, filename)
        df = pd.read_csv(file_path, encoding='utf-8')
        df['时间'] = pd.to_datetime(df['时间'])
        df.set_index('时间', inplace=True)

        # 数据清洗：移除包含NaN的行
        df = df.dropna(subset=['收盘价', '成交额.元.'])

        # 检查价格是否为零或负值
        if (df['收盘价'] <= 0).any():
            print(f"警告：股票 {stock_code} 包含非正价格，跳过此股票")
            continue

        stock_data[stock_code] = df[['收盘价', '成交额.元.']]

# 3. 初始化策略类
# 可以调整策略参数
strategy = PairTradingStrategy(
    stock_data,
    industry_df,
    formation_period=252,  # 形成期长度
    z_threshold=1.0,  # Z得分阈值
    stop_loss=3.0,  # 止损阈值
    take_profit=0.2,  # 止盈阈值（10%收益）
    initial_capital=1000000  # 初始资金
)

# 4. 指定股票对
# 这里可以修改为你想要的股票对
# 格式：[(股票1代码, 股票2代码), (股票3代码, 股票4代码), ...]
specified_pairs = [
    # ('603596.SH', '603129.SH'),  # 示例股票对1
    ('600295.SH', '000825.SZ')  # 示例股票对2
]

# 5. 处理指定的股票对（取消协整检验）
print("正在处理指定的股票对...")
for pair in specified_pairs:
    s1, s2 = pair

    # 检查股票是否存在于数据中
    if s1 not in stock_data:
        print(f"警告：股票 {s1} 不存在于数据中，跳过此股票对")
        continue
    if s2 not in stock_data:
        print(f"警告：股票 {s2} 不存在于数据中，跳过此股票对")
        continue

    # 获取形成期数据
    if len(stock_data[s1]) < strategy.formation_period or len(stock_data[s2]) < strategy.formation_period:
        print(f"警告：股票 {s1} 或 {s2} 的数据不足，跳过此股票对")
        continue

    s1_data = stock_data[s1].iloc[:strategy.formation_period]
    s2_data = stock_data[s2].iloc[:strategy.formation_period]

    # 确保两个股票的日期完全对齐
    common_dates = s1_data.index.intersection(s2_data.index)
    if len(common_dates) < strategy.formation_period * 0.9:  # 要求至少90%的日期匹配
        print(f"警告：股票 {s1} 和 {s2} 的日期匹配度不足，跳过此股票对")
        continue

    s1_data = s1_data.loc[common_dates]
    s2_data = s2_data.loc[common_dates]

    # 计算价差
    spread = s1_data['收盘价'] - s2_data['收盘价']

    # 检查价差序列是否有效
    if spread.std() < 1e-10:  # 价差几乎恒定
        print(f"警告：股票对 {pair} 的价差几乎恒定，跳过此股票对")
        continue

    if np.isnan(spread).any() or np.isinf(spread).any():
        print(f"警告：股票对 {pair} 的价差包含NaN或inf，跳过此股票对")
        continue

    # 计算价差统计特性
    mean = spread.mean()
    std = spread.std()

    # 计算半衰期（增加异常处理）
    try:
        half_life = strategy._calculate_half_life(spread)
        if np.isnan(half_life) or np.isinf(half_life) or half_life <= 0:
            raise ValueError("无效的半衰期值")
    except Exception as e:
        print(f"警告：计算股票对 {pair} 的半衰期时出错: {str(e)}，使用默认值20天")
        half_life = 20  # 默认半衰期值

    # 计算日均成交额
    volume_s1 = s1_data['成交额.元.']
    volume_s2 = s2_data['成交额.元.']
    avg_volume_s1 = volume_s1.mean()
    avg_volume_s2 = volume_s2.mean()
    total_avg_volume = avg_volume_s1 + avg_volume_s2

    strategy.selected_pairs.append({
        'pair': pair,
        'spread_mean': mean,
        'spread_std': std,
        'half_life': half_life,
        'total_avg_volume': total_avg_volume
    })

    print(f"已添加股票对 {pair}，价差标准差: {std:.4f}，半衰期: {half_life:.2f}天")

# 6. 检查是否有有效的股票对
if not strategy.selected_pairs:
    print("没有找到有效的股票对进行回测，请检查指定的股票对和数据")
    exit()

# 7. 输出选择的股票对
print("\n最终选择的股票对：")
for i, p in enumerate(strategy.selected_pairs):
    print(f"{i + 1}. 股票对: {p['pair']}, 价差标准差: {p['spread_std']:.4f}, 日均成交额: {p['total_avg_volume']:.2f}元")

# 8. 回测策略
print(f"\n开始回测，日期范围: {args.start_date} 至 {args.end_date}...")
start_date = pd.to_datetime(args.start_date)
end_date = pd.to_datetime(args.end_date)
performance = strategy.backtest(start_date, end_date)

# 9. 输出回测结果
print("\n策略表现：")
for key, value in performance.items():
    if key == '夏普比率' or key == '最大回撤':
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value:.2%}")

# 10. 为每个股票对绘制组合价值与Z得分曲线
dates, values = zip(*strategy.portfolio_value)

# 设置中文字体
plt.rcParams["font.family"] = ["Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
def format_currency(x, pos):
    return f'{int(x):,}'
# 绘制每个股票对的组合价值与Z得分图
for i, pair_info in enumerate(strategy.selected_pairs):
    pair = pair_info['pair']
    s1, s2 = pair

    # 获取回测期间的Z得分数据
    pair_dates = []
    pair_z_scores = []

    for date, _ in strategy.portfolio_value:
        if date in stock_data[s1].index and date in stock_data[s2].index:
            price1 = stock_data[s1].loc[date, '收盘价']
            price2 = stock_data[s2].loc[date, '收盘价']
            spread = price1 - price2
            z_score = (spread - pair_info['spread_mean']) / pair_info['spread_std']

            pair_dates.append(date)
            pair_z_scores.append(z_score)

    # 创建图表
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # 绘制组合价值变化（左侧Y轴）
    line1 = ax1.plot(dates, values, label='组合价值', color='green')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('组合价值 (元)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.yaxis.set_major_formatter(FuncFormatter(format_currency))
    # 设置x轴日期格式
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()  # 自动旋转日期标签

    # 添加网格线
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 创建第二个Y轴用于Z得分（右侧Y轴）
    ax2 = ax1.twinx()

    # 绘制Z得分曲线
    line2 = ax2.plot(pair_dates, pair_z_scores, label=f'{pair} Z得分', color='grey', linestyle='--')
    ax2.set_ylabel('Z得分', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # 添加Z得分阈值线
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.axhline(y=strategy.z_threshold, color='red', linestyle='--', label='上阈值')
    ax2.axhline(y=-strategy.z_threshold, color='green', linestyle='--', label='下阈值')
    # ax2.axhline(y=strategy.stop_loss, color='purple', linestyle='-.', label='止损阈值')
    # ax2.axhline(y=-strategy.stop_loss, color='purple', linestyle='-.')

    # 添加图例
    lines = line1 + line2
    labels = ['组合价值', f'Z得分']
    ax1.legend(lines, labels, loc='upper left')

    # 添加标题
    plt.title(f'汽车行业配对交易策略资产组合价值与 {pair} Z得分曲线 ({args.start_date}至{args.end_date})')

    # 保存组合价值与Z得分图
    pair_img_path = os.path.join(args.output_dir, f'portfolio_{s1}_{s2}_zscore_{args.start_date}_{args.end_date}.png')
    plt.tight_layout()
    plt.savefig(pair_img_path, dpi=300, bbox_inches='tight')
    print(f"股票对 {pair} 的组合价值与Z得分图已保存至: {pair_img_path}")
    plt.close()  # 关闭当前图表，避免内存占用

# 11. 保存回测结果到CSV（去除累积收益率相关列）
results = {
    '日期': [d.strftime('%Y-%m-%d') for d, _ in strategy.portfolio_value],
    '组合价值': [v for _, v in strategy.portfolio_value]
}

results_df = pd.DataFrame(results)
results_csv_path = os.path.join(args.output_dir, f'backtest_results_{args.start_date}_{args.end_date}.csv')
results_df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')
print(f"回测结果已保存至: {results_csv_path}")