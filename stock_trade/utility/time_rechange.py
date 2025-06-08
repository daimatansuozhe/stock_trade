import os
import pandas as pd
from datetime import datetime

def filter_csv_by_date(input_file, output_file, target_date_str='2024-01-01'):
    """
    过滤CSV文件，保留时间列中日期大于或等于目标日期的行

    参数:
    input_file (str): 输入CSV文件路径
    output_file (str): 输出CSV文件路径
    target_date_str (str): 目标日期字符串，格式为 'YYYY-MM-DD'
    """
    # 将目标日期字符串转换为datetime对象
    target_date = datetime.strptime(target_date_str, '%Y-%m-%d')

    try:
        # 读取CSV文件
        df = pd.read_csv(input_file)

        # 查找时间列（处理可能的列名变体）
        time_col = None
        expected_names = ['时间', '"时间"', '时间 ', '"时间" ']

        for col in df.columns:
            if col.strip('" ') in [name.strip('" ') for name in expected_names]:
                time_col = col
                break

        if time_col is None:
            print(f"警告：文件 '{input_file}' 中未找到时间列，已跳过。")
            return False

        # 转换时间列为datetime类型
        # 尝试多种日期格式解析
        success = False
        date_formats = ['%Y-%m-%d', '%Y/%m/%d', '%Y年%m月%d日', '%Y-%m-%d %H:%M:%S']

        for fmt in date_formats:
            try:
                df[time_col] = pd.to_datetime(df[time_col], format=fmt)
                success = True
                break
            except ValueError:
                continue

        if not success:
            print(f"警告：文件 '{input_file}' 中的时间格式无法识别。")
            return False

        # 过滤数据，保留大于或等于目标日期的行
        filtered_df = df[df[time_col] >= target_date]

        # 如果过滤后没有数据，输出警告
        if filtered_df.empty:
            print(f"警告：文件 '{input_file}' 中没有2024-03-01及以后的数据。")

        # 保存过滤后的数据到新文件
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        filtered_df.to_csv(output_file, index=False)

        print(f"成功处理文件: {input_file} -> {output_file}")
        return True

    except Exception as e:
        print(f"处理文件 '{input_file}' 时出错: {e}")
        return False

def batch_process_csv_files(input_dir, output_dir, target_date_str='2024-01-01'):
    """
    批量处理目录中的所有CSV文件

    参数:
    input_dir (str): 输入目录路径
    output_dir (str): 输出目录路径
    target_date_str (str): 目标日期字符串，格式为 'YYYY-MM-DD'
    """
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误：输入目录 '{input_dir}' 不存在。")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)

            # 处理单个CSV文件
            filter_csv_by_date(input_file, output_file, target_date_str)

if __name__ == "__main__":
    # 设置输入和输出目录路径，请修改为实际路径
    input_directory = './raw_data_zz500成分股日数据'
    output_directory = './data_trading'

    # 批处理CSV文件
    batch_process_csv_files(input_directory, output_directory)

    print("批量处理完成！")