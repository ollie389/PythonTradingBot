import datetime as dt
import pytz
import plotly.graph_objects as go
import plotly.express as px
import talib as ta
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd
import Crypto_Strategy_Kmean
import ccxt
import calendar
import time as time_module
import os
from multiprocessing import Process, Manager


class Data:
    def __init__(self, dataframe):
        self.df = dataframe
        self.misc = {}
        self.sr_long = {}
        self.sr_short = {}
        self.support = {}
        self.resistance = {}
        self.support_plot = {}
        self.resistance_plot = {}
        self.centers_plot = {}

    def update(self, dataframe):
        self.df = dataframe



def fetch_historical_data(exchange, symbol, timeframe, from_ts, limit):
    ohlcv_data = []
    while True:
        new_ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=from_ts, limit=limit)
        if not new_ohlcv:
            break
        ohlcv_data.extend(new_ohlcv)
        if len(new_ohlcv) < limit:
            break
        from_ts = new_ohlcv[-1][0]
    return ohlcv_data


def initialize_bars(ohlcv):
    bars = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    bars['time'] = pd.to_datetime(bars['time'], unit='ms')
    bars.set_index('time', inplace=True)
    return bars


def add_extra_columns(bars):
    columns = [
        'buy', 'short', 'sell_price', 'sell_price_s',
        'stop_loss', 'stop_loss2', 'stop_loss_s', 'stop_loss_s2',
        'hold', 'hold_s', 'sell', 'sell_s', 'balance', 'balance_s',
        'buy_price', 'buy_price_s'
    ]
    for col in columns:
        bars[col] = np.nan if 'price' in col else 0
    bars['balance'] = 10000
    bars['balance_s'] = 10000
    return bars


def update_balance(bars, hold, hold_s, balance, balance_old, long_profit, short_profit):
    if hold:
        balance = hold * bars['close'][-1]
        long_profit += balance - balance_old
    if hold_s:
        balance += hold_s * (bars['buy_price_s'] - bars['close'][-1])
        short_profit += balance - balance_old
    return balance, long_profit, short_profit


def run_backtest(symbol, symbols, ns, results):
    print(f'Starting backtest on {symbol}', flush=True)

    strategy = Crypto_Strategy_Kmean.strategy
    starting_balance = 10000
    balance = starting_balance
    balance_old = starting_balance

    # Fetch historical data
    exchange = ccxt.bybit()
    from_ts = exchange.parse8601('2023-11-01 00:00:00')
    ohlcv = fetch_historical_data(exchange, symbol, '5m', from_ts, limit=200)
    bars = initialize_bars(ohlcv)
    bars = add_extra_columns(bars)

    data = Data(bars)
    data.update(bars)

    short_profit = 0
    long_profit = 0
    hold = 0
    hold_s = 0

    for j in range(1, len(bars)):
        # Update strategy
        data = strategy(data)
        bars = data.df

        # Update balance and profits
        balance, long_profit, short_profit = update_balance(bars, hold, hold_s, balance, balance_old, long_profit, short_profit)
        balance_old = balance

        # Log results
        if j == len(bars) - 1:
            results[symbol].append(balance - starting_balance)
            results_dict = dict(results)
            df = pd.concat({k: pd.Series(v) for k, v in results_dict.items()}, axis=1)
            df.to_csv('out.csv')

        print(f'{symbol}: {int(balance)}, long: {int(long_profit)}, short: {int(short_profit)}')
        try:
            print(f'balance: {balance}, win: {ns.win}, lose: {ns.lose}, wl ratio: {ns.win / (ns.win + ns.lose)}')
        except ZeroDivisionError:
            print('no trades yet')



def plot_trade(data, data_sim, last_bought, j, test):
    try:
        sr_search_range = 200
        sr_buffer = 15

        # Prepare the data
        bars = data.df
        end = len(bars) - 1
        start_index = last_bought - (sr_search_range + sr_buffer)
        end_index = min(j + 100, end)

        # Create the candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=bars.index[start_index:end_index],
            open=bars['open'][start_index:end_index],
            high=bars['high'][start_index:end_index],
            low=bars['low'][start_index:end_index],
            close=bars['close'][start_index:end_index]
        )])

        # Determine support and resistance centers
        if test == 0:
            centers_long = data_sim.sr_long[last_bought - bars.turningpoints[last_bought]]
            support = []
            resistance = []
        else:
            centers_long = data.centers_plot[last_bought]
            support = data.support[last_bought]
            resistance = data.resistance[last_bought]

        # Add support and resistance areas
        add_support_resistance_areas(fig, bars, support, resistance, start_index, end_index, sr_search_range, sr_buffer)

        # Add moving averages and other indicators
        add_indicators(fig, bars, last_bought, start_index, end_index, sr_search_range, sr_buffer)

        # Plot supertrend
        plot_supertrend(fig, bars, last_bought, start_index, end_index, sr_search_range, sr_buffer)

        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.write_html(f"images/fig{j}.html")

    except Exception as e:
        print(f"An error occurred: {e}")


def add_support_resistance_areas(fig, bars, support, resistance, start_index, end_index, sr_search_range, sr_buffer):
    for center in support:
        fig.add_trace(go.Scatter(
            x=[bars.index[start_index], bars.index[start_index], bars.index[end_index], bars.index[end_index], bars.index[start_index]],
            y=[center[0], center[1], center[1], center[0], center[0]],
            fill="toself", opacity=0.25, mode="lines", fillcolor="Green", line=dict(color="Green")
        ))
    for center in resistance:
        fig.add_trace(go.Scatter(
            x=[bars.index[start_index], bars.index[start_index], bars.index[end_index], bars.index[end_index], bars.index[start_index]],
            y=[center[0], center[1], center[1], center[0], center[0]],
            fill="toself", opacity=0.25, mode="lines", fillcolor="Red", line=dict(color="Red")
        ))
    for center in data.centers_plot[last_bought]:
        fig.add_trace(go.Scatter(
            x=[bars.index[start_index], bars.index[start_index], bars.index[end_index], bars.index[end_index], bars.index[start_index]],
            y=[center[0], center[1], center[1], center[0], center[0]],
            fill="toself", opacity=0.25, mode="lines", fillcolor="Blue", line=dict(color="Blue")
        ))

    fig.add_vline(x=bars.index[last_bought - sr_buffer], line=dict(color='Black', dash='dash'), opacity=0.25)
    fig.add_vline(x=bars.index[last_bought], opacity=0.25)


def add_indicators(fig, bars, last_bought, start_index, end_index, sr_search_range, sr_buffer):
    sma_low = (bars.sma_high + bars.sma_low) / 2
    sma_high = sma_low
    if bars['low'][last_bought] > bars['ema'][last_bought]:
        fig.add_trace(go.Scatter(
            x=bars.index[start_index:last_bought],
            y=sma_low[start_index:last_bought],
            mode='lines', line=dict(color="Blue")
        ))
    else:
        fig.add_trace(go.Scatter(
            x=bars.index[start_index:last_bought],
            y=sma_high[start_index:last_bought],
            mode='lines', line=dict(color="Blue")
        ))

    fig.add_vline(x=bars.index[int(last_bought - bars.turningpoints[last_bought])], opacity=0.25)
    fig.add_vline(x=bars.index[end_index - 1], opacity=0.25)

    fig.add_trace(go.Scatter(
        x=bars.index[last_bought - sr_buffer:last_bought], y=bars.upperband[last_bought - sr_buffer:last_bought],
        mode='lines', line=dict(color="#2f98de")
    ))
    fig.add_trace(go.Scatter(
        x=bars.index[last_bought - sr_buffer:last_bought], y=bars.lowerband[last_bought - sr_buffer:last_bought],
        mode='lines', line=dict(color="#2f98de")
    ))
    fig.add_trace(go.Scatter(
        x=bars.index[start_index:end_index], y=bars.ema[start_index:end_index],
        mode='lines', line=dict(color="#ff9933")
    ))
    fig.add_trace(go.Scatter(
        x=bars.index[start_index:end_index], y=bars.ema2[start_index:end_index],
        mode='lines', line=dict(color="#682390")
    ))
    fig.add_trace(go.Scatter(
        x=bars.index[start_index:end_index], y=bars.ema3[start_index:end_index],
        mode='lines', line=dict(color="#924f9b")
    ))


def plot_supertrend(fig, bars, last_bought, start_index, end_index, sr_search_range, sr_buffer):
    df = pd.DataFrame()
    df.index = bars.index[start_index:end_index]
    df['ma1'] = (bars['high'][start_index:end_index] + bars['low'][start_index:end_index]) / 2
    df['ma2'] = bars['supertrend'][start_index:end_index]
    df['label'] = np.where(df['ma1'] > df['ma2'], 1, 0)
    df['group'] = df['label'].ne(df['label'].shift()).cumsum()
    df = df.groupby('group')

    dfs = [data for _, data in df]

    def fillcol(label):
        return 'rgba(0,250,0,0.1)' if label >= 1 else 'rgba(250,0,0,0.1)'

    def fillcol_solid(label):
        return 'rgba(0,250,0,0.4)' if label >= 1 else 'rgba(250,0,0,0.4)'

    for df in dfs:
        fig.add_trace(go.Scatter(
            x=df.index, y=df.ma1, mode='lines', line=dict(color=fillcol(df['label'].iloc[0]))
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df.ma2, mode='lines', line=dict(color=fillcol_solid(df['label'].iloc[0])),
            fill='tonexty', fillcolor=fillcol(df['label'].iloc[0])
        ))

if __name__ == "__main__":
    if not os.path.exists("images"):
        os.mkdir("images")

    symbols = ['BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'SOLUSDT', 'DOTUSDT', 'LINKUSDT', 'ETCUSDT', 'XRPUSDT']

    mgr = Manager()
    ns = mgr.Namespace()
    results = mgr.dict()
    ns.win = 0
    ns.lose = 0

    for s in symbols:
        results[s] = mgr.list()

    processes = [Process(target=run_backtest, args=(s, symbols, ns, results)) for s in symbols]

    for process in processes:
        process.start()
    for process in processes:
        process.join()



