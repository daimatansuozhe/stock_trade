import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


class PairTradingStrategy:
    def __init__(self, stock_data, industry_df, formation_period=252, z_threshold=0.1,
                 stop_loss=3.0, take_profit=-1.0, initial_capital=1000000):
        """
        初始化配对交易策略类

        参数:
        stock_data: 包含所有股票数据的字典，键为股票代码，值为DataFrame
        industry_df: 行业分类数据的DataFrame
        formation_period: 形成期长度（交易日数）
        z_threshold: Z得分阈值，用于触发交易信号
        stop_loss: 止损阈值
        take_profit: 止盈阈值（负值表示不使用止盈）
        initial_capital: 初始资金
        """
        self.stock_data = stock_data
        self.industry_df = industry_df
        self.formation_period = formation_period
        self.z_threshold = z_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.initial_capital = initial_capital
        self.selected_pairs = []
        self.portfolio_value = []

    def select_pairs(self):
        """自动选择符合条件的股票对"""
        # 这里省略自动选择股票对的实现
        return self.selected_pairs

    def _cointegration_test(self, series1, series2):
        """进行协整检验"""
        _, pvalue, _ = coint(series1, series2)
        return pvalue

    def _calculate_half_life(self, spread):
        """计算价差的半衰期"""
        spread_lag = spread.shift(1)
        spread_lag.iloc[0] = spread_lag.iloc[1]

        spread_ret = spread - spread_lag
        spread_ret.iloc[0] = spread_ret.iloc[1]

        spread_lag2 = np.vstack([spread_lag])
        spread_lag2 = np.reshape(spread_lag2, (len(spread_lag2[0]), 1))

        model = np.linalg.lstsq(spread_lag2, spread_ret, rcond=None)
        lambda_ = model[0][0]

        half_life = -np.log(2) / lambda_
        return half_life

    def backtest(self, start_date, end_date):
        """
        回测策略

        参数:
        start_date: 回测开始日期
        end_date: 回测结束日期
        """
        # 初始化资金和持仓
        capital = self.initial_capital
        positions = {}

        # 获取所有股票在回测期间的共同日期
        all_dates = set()
        for df in self.stock_data.values():
            dates = df.index[(df.index >= start_date) & (df.index <= end_date)]
            all_dates.update(dates)

        all_dates = sorted(all_dates)

        # 记录每日组合价值
        daily_value = []

        for date in all_dates:
            # 检查是否所有股票都有该日期的数据
            valid_pairs = []
            for pair_info in self.selected_pairs:
                pair = pair_info['pair']
                s1, s2 = pair
                if s1 in self.stock_data and s2 in self.stock_data:
                    if date in self.stock_data[s1].index and date in self.stock_data[s2].index:
                        valid_pairs.append(pair_info)

            # 如果没有有效的股票对，跳过该日期
            if not valid_pairs:
                continue

            # 计算组合价值
            portfolio_value = capital
            for pair_info in valid_pairs:
                pair = pair_info['pair']
                s1, s2 = pair
                if pair in positions:
                    pos1, pos2 = positions[pair]
                    price1 = self.stock_data[s1].loc[date, '收盘价']
                    price2 = self.stock_data[s2].loc[date, '收盘价']
                    portfolio_value += pos1 * price1 + pos2 * price2

            daily_value.append((date, portfolio_value))

            # 交易信号生成和执行
            for pair_info in valid_pairs:
                pair = pair_info['pair']
                s1, s2 = pair
                mean = pair_info['spread_mean']
                std = pair_info['spread_std']

                # 获取当前价格
                price1 = self.stock_data[s1].loc[date, '收盘价']
                price2 = self.stock_data[s2].loc[date, '收盘价']

                # 计算当前价差和Z得分
                current_spread = price1 - price2
                z_score = (current_spread - mean) / std

                # 检查是否已有持仓
                if pair in positions:
                    pos1, pos2 = positions[pair]

                    # 计算持仓价值和初始价值
                    position_value = pos1 * price1 + pos2 * price2

                    # 计算持仓收益率（使用首次开仓时的价格）
                    initial_price1 = self.stock_data[s1].loc[all_dates[0], '收盘价']
                    initial_price2 = self.stock_data[s2].loc[all_dates[0], '收盘价']
                    initial_value = pos1 * initial_price1 + pos2 * initial_price2

                    # 计算持仓收益率
                    position_return = (position_value / initial_value) - 1

                    # 止损条件
                    if abs(z_score) >= self.stop_loss:
                        capital += position_value
                        positions.pop(pair)
                        print(f"{date}: 对 {pair} 执行止损，Z得分: {z_score:.4f}")

                    # 止盈条件
                    elif self.take_profit > 0 and position_return >= self.take_profit:
                        capital += position_value
                        positions.pop(pair)
                        print(f"{date}: 对 {pair} 执行止盈，收益率: {position_return:.4f}")

                    # 平仓条件（价差回归）
                    elif (pos1 > 0 and z_score <= 0) or (pos1 < 0 and z_score >= 0):
                        capital += position_value
                        positions.pop(pair)
                        print(f"{date}: 对 {pair} 执行平仓，Z得分: {z_score:.4f}")

                # 开仓条件
                else:
                    if z_score >= self.z_threshold:
                        # 做空股票1，做多股票2
                        # 计算分配给每个股票对的资金
                        pair_capital = capital / len(valid_pairs)

                        # 计算可以买入的股票数量
                        shares1 = int(-pair_capital / 2 / price1)
                        shares2 = int(pair_capital / 2 / price2)

                        # 更新资金和持仓
                        capital += shares1 * price1 + shares2 * price2
                        positions[pair] = (shares1, shares2)
                        print(f"{date}: 开仓 {pair}，做空 {s1} {shares1} 股，做多 {s2} {shares2} 股，Z得分: {z_score:.4f}")

                    elif z_score <= -self.z_threshold:
                        # 做多股票1，做空股票2
                        pair_capital = capital / len(valid_pairs)
                        shares1 = int(pair_capital / 2 / price1)
                        shares2 = int(-pair_capital / 2 / price2)

                        capital += shares1 * price1 + shares2 * price2
                        positions[pair] = (shares1, shares2)
                        print(f"{date}: 开仓 {pair}，做多 {s1} {shares1} 股，做空 {s2} {shares2} 股，Z得分: {z_score:.4f}")

        # 回测结束，平仓所有持仓
        for pair, (pos1, pos2) in positions.items():
            s1, s2 = pair
            if date in self.stock_data[s1].index and date in self.stock_data[s2].index:
                price1 = self.stock_data[s1].loc[date, '收盘价']
                price2 = self.stock_data[s2].loc[date, '收盘价']
                capital += pos1 * price1 + pos2 * price2

        # 计算回测结果
        total_return = (capital / self.initial_capital) - 1
        daily_returns = [(daily_value[i][1] / daily_value[i - 1][1] - 1) for i in range(1, len(daily_value))]
        annual_return = np.mean(daily_returns) * 252
        annual_volatility = np.std(daily_returns) * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0

        # 计算最大回撤
        peak = daily_value[0][1]
        max_drawdown = 0
        for date, value in daily_value:
            if value > peak:
                peak = value
            drawdown = (value / peak) - 1
            if drawdown < max_drawdown:
                max_drawdown = drawdown

        # 保存组合价值序列
        self.portfolio_value = daily_value

        return {
            '总收益率': total_return,
            '年化收益率': annual_return,
            '年化波动率': annual_volatility,
            '夏普比率': sharpe_ratio,
            '最大回撤': max_drawdown
        }