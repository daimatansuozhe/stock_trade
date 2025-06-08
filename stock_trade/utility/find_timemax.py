import os
import pandas as pd
from datetime import datetime


def find_max_time_in_first_rows(directory_path):
    """
    遍历目录中所有CSV文件，读取每个文件的第一行数据（不含列名），
    并返回其中时间列的最大值及其所在的文件路径
    """
    max_time = None
    max_time_file = None

    # 检查目录是否存在
    if not os.path.exists(directory_path):
        print(f"错误：目录 '{directory_path}' 不存在。")
        return None, None

    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)

            try:
                # 读取CSV文件的第一行数据（跳过列名）
                df = pd.read_csv(file_path, nrows=1)

                # 获取第一行数据
                first_row = df.iloc[0]

                # 查找时间列（处理可能的列名变体）
                time_value = None
                expected_names = ['时间', '"时间"', '时间 ', '"时间" ']

                for col in df.columns:
                    if col.strip('" ') in [name.strip('" ') for name in expected_names]:
                        time_value = first_row[col]
                        break

                if time_value is None:
                    print(f"警告：文件 '{filename}' 中未找到时间列，已跳过。")
                    continue

                # 尝试解析时间值
                try:
                    # 处理带引号的时间字符串
                    if isinstance(time_value, str):
                        time_value = time_value.strip('"')

                    # 尝试多种日期格式解析
                    date_formats = ['%Y-%m-%d', '%Y/%m/%d', '%Y年%m月%d日']
                    parsed_time = None

                    for fmt in date_formats:
                        try:
                            parsed_time = datetime.strptime(time_value, fmt)
                            break
                        except ValueError:
                            continue

                    if parsed_time is None:
                        print(f"警告：文件 '{filename}' 中的时间格式无法识别: {time_value}")
                        continue

                    # 更新最大时间和对应文件
                    if max_time is None or parsed_time > max_time:
                        max_time = parsed_time
                        max_time_file = file_path

                except Exception as e:
                    print(f"警告：解析文件 '{filename}' 的时间值时出错: {e}")
                    continue

            except Exception as e:
                print(f"处理文件 '{filename}' 时出错: {e}")
                continue

    return max_time, max_time_file


if __name__ == "__main__":
    # 设置CSV文件所在目录路径，请修改为实际路径
    directory_path = '../raw_data_zz500成分股日数据'

    # 查找最大时间及其所在文件
    max_time, max_time_file = find_max_time_in_first_rows(directory_path)

    # 输出结果
    if max_time is not None and max_time_file is not None:
        print(f"所有CSV文件第一行数据中的最大时间值是: {max_time.strftime('%Y-%m-%d')}")
        print(f"最大值所在的文件: {max_time_file}")
    else:
        print("未能找到有效的最大时间值。")