import sys
import ccxt
import pandas as pd
import Crypto_Strategy_Kmean
import datetime as dt
import numpy as np
import time
import plotly.graph_objects as go
from pybit import unified_trading
from config import SYMBOLS, BALANCE_ALLOCATION, DATA_WINDOW, TIMEFRAME, TIMEFRAME_SEC, SL_RATIO, TP_RATIO, LEVERAGE
from config import STRATEGY as strategy

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


def clear_all_orders(exchange):
    for symbol in SYMBOLS:
        orders = exchange.fetch_open_orders(symbol)
        for order in orders:
            exchange.cancel_order(order['id'], symbol=symbol)


def clear_all_positions(exchange):
    for symbol in SYMBOLS:
        positions = exchange.fetch_positions(symbols=[symbol])
        for position in positions:
            symbol = position['symbol']
            size = position['contracts']
            side = position['info']['side']
            if side == 'Buy' and size > 0:
                exchange.create_market_sell_order(symbol, size, params={'position_idx': 0})
            elif size > 0:
                exchange.create_market_buy_order(symbol, size, params={'position_idx': 0})


def gen_signals(exchange, symbols, data_window, timeframe, from_ts):
    data_dict = {}
    for symbol in symbols:
        ohlcv = fetch_ohlcv_data(exchange, symbol, timeframe, from_ts, data_window)
        data = Data(pd.DataFrame(ohlcv))
        bars = data.df
        bars.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        bars['time'] = pd.to_datetime(bars['time'], unit='ms')
        bars.set_index('time', inplace=True)
        bars = add_extra_columns(bars)
        data = strategy(data, stream=0)
        data_dict[symbol] = data
    return data_dict


def fetch_ohlcv_data(exchange, symbol, timeframe, from_ts, data_window):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=from_ts, limit=data_window)
    new_ohlcv = ohlcv
    while len(new_ohlcv) == 200:
        from_ts_new = ohlcv[-1][0]
        new_ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=from_ts_new, limit=data_window)
        ohlcv.extend(new_ohlcv)
    return ohlcv


def add_extra_columns(bars):
    columns = [
        'buy', 'short', 'sell_price', 'sell_price_s',
        'stop_loss', 'stop_loss2', 'stop_loss_s', 'stop_loss_s2',
        'hold', 'hold_s', 'sell', 'sell_s', 'buy_price', 'buy_price_s'
    ]
    for col in columns:
        bars[col] = np.nan if 'price' in col else 0
    return bars


def update_signals(exchange, data_dict, timeframe, timeframe_sec):
    updated = 0
    new_data = 0
    since = exchange.milliseconds() - timeframe_sec * 1000
    for symbol in SYMBOLS:
        length = len(data_dict[symbol].df)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=2)
        if len(ohlcv) > 0:
            ohlcv_time = dt.datetime.fromtimestamp(float(ohlcv[-1][0]) / 1000)
            if data_dict[symbol].df.index[-1] != ohlcv_time:
                bars = data_dict[symbol].df
                row_copy = bars.iloc[-1].copy()
                bars.loc[ohlcv_time] = row_copy * np.nan
                bars.iloc[-1, bars.columns.get_loc('open')] = ohlcv[-1][1]
                bars.iloc[-1, bars.columns.get_loc('high')] = ohlcv[-1][2]
                bars.iloc[-1, bars.columns.get_loc('low')] = ohlcv[-1][3]
                bars.iloc[-1, bars.columns.get_loc('close')] = ohlcv[-1][4]
                bars.iloc[-1, bars.columns.get_loc('volume')] = ohlcv[-1][5]
                bars.drop(bars.index[0], inplace=True)
                data = strategy(data_dict[symbol], data_window=DATA_WINDOW, stream=1)
                data_dict[symbol] = data
                updated = 1
                new_data = 1
    if updated:
        print('updated')
        sys.stdout.flush()
    return [data_dict, new_data]


def run_search(queue1, data_dict, exchange):
    while True:
        new_data = 0
        refresh_time = 0
        min = dt.datetime.now().minute
        while refresh_time == 0:
            if dt.datetime.now().minute > min:
                min = dt.datetime.now().minute
            time.sleep(1)
            now = dt.datetime.now()
            if now.minute % (TIMEFRAME_SEC / 60) == (TIMEFRAME_SEC / 60 - 1):
                if now.second > 58:
                    refresh_time = 1
                    break

        while new_data == 0:
            data_dict, new_data = update_signals(exchange, data_dict, TIMEFRAME, TIMEFRAME_SEC)
            if not new_data:
                time.sleep(1)
            else:
                queue1.put(data_dict)


def set_leverage(leverage, exchange, symbols):
    for symbol in symbols:
        position = exchange.fetch_positions(symbols=[symbol])[0]
        if int(position['info']['leverage']) != leverage:
            exchange.set_leverage(leverage, symbol=symbol, params={})


def adjust_leverage(balance, starting_balance, lev):
    if balance <= starting_balance * (1 - 1 * BALANCE_ALLOCATION):
        return int(starting_balance / balance)
    return 1


def set_limit_order(exchange, position, symbol, size, side, position_strat):
    if position_strat[symbol] == 0:
        if side == 'Buy':
            take_profit = float(position['info']['take_profit']) / 1.01
            side = 'Sell'
        else:
            side = 'Buy'
            take_profit = float(position['info']['take_profit']) / 0.99
        params = {'position_idx': 0, 'reduce_only': 'true'}
        print('Setting limit order on %s' % symbol)
        exchange.create_order(symbol, 'limit', side, size, take_profit, params)

        stop_loss = set_stop_loss(position, side)
        params['triggerPrice'] = stop_loss
        print('Setting limit order on %s' % symbol)
        exchange.create_order(symbol, 'limit', side, size, stop_loss, params)


def set_stop_loss(position, side):
    if side == 'Buy':
        return float(position['info']['stop_loss']) / 0.99
    return float(position['info']['stop_loss']) / 1.01


def handle_trailing_stop(exchange, data, position, symbol, size, side, entry_price, position_HL, position_trail):
    new_trail = 0
    if side == 'Buy':
        position_HL[symbol] = max(position_HL[symbol], data.df['high'][-1])
        trail_length = max((position_HL[symbol] - entry_price[symbol]) / 3, (entry_price[symbol] - position_trail[symbol]))
        if position_HL[symbol] - trail_length > entry_price[symbol]:
            result = exchange.fetch_order_book(symbol, limit=2)
            bid = result['bids'][0][0]
            new_trail = min(position_HL[symbol] - trail_length, bid)
        side = 'Sell'
    else:
        position_HL[symbol] = min(position_HL[symbol], data.df['low'][-1])
        trail_length = max((entry_price[symbol] - position_HL[symbol]) / 3, (position_trail[symbol] - entry_price[symbol]))
        if position_HL[symbol] + trail_length < entry_price[symbol]:
            result = exchange.fetch_order_book(symbol, limit=2)
            ask = result['asks'][0][0]
            new_trail = max(position_HL[symbol] + trail_length, ask)
        side = 'Buy'
    if new_trail > 0:
        update_trailing_stop(exchange, symbol, size, side, new_trail)


def update_trailing_stop(exchange, symbol, size, side, new_trail):
    params = {'position_idx': 0, 'reduce_only': 'true', 'triggerPrice': new_trail}
    orders = exchange.fetch_open_orders(symbol)
    id = orders[0]['id']
    exchange.cancel_order(id, symbol=symbol)
    exchange.create_order(symbol, 'limit', side, size, new_trail, params)
    print('Setting trailing stop limit order on %s' % symbol)


def handle_new_signals(exchange, data, symbol, entry_price, position_strat, position_trail, in_position, balance, starting_balance):
    if (data.df['buy'][-1] == 1 or data.df['short'][-1] == 1) and len(exchange.fetch_open_orders(symbol)) == 0:
        if data.df['buy'][-1] == 1 and not in_position:
            handle_buy_signal(exchange, data, symbol, entry_price, position_strat, balance, starting_balance)
        elif data.df['short'][-1] == 1 and not in_position:
            handle_short_signal(exchange, data, symbol, entry_price, position_strat, balance, starting_balance)


def handle_buy_signal(exchange, data, symbol, entry_price, position_strat, balance, starting_balance):
    leverage = adjust_leverage(balance, starting_balance, LEVERAGE)
    set_position_leverage(exchange, symbol, leverage)
    result = exchange.fetch_order_book(symbol, limit=2)
    bid = result['bids'][0][0]
    params = get_order_params(data, symbol, position_strat, 'buy')
    print('Buying %s' % symbol)
    exchange.create_order(symbol, 'limit', 'buy', starting_balance * BALANCE_ALLOCATION / bid, bid, params)
    entry_price[symbol] = bid


def handle_short_signal(exchange, data, symbol, entry_price, position_strat, balance, starting_balance):
    leverage = adjust_leverage(balance, starting_balance, LEVERAGE)
    set_position_leverage(exchange, symbol, leverage)
    result = exchange.fetch_order_book(symbol, limit=2)
    ask = result['asks'][0][0]
    params = get_order_params(data, symbol, position_strat, 'short')
    print('Shorting %s' % symbol)
    exchange.create_order(symbol, 'limit', 'sell', starting_balance * BALANCE_ALLOCATION / ask, ask, params)
    entry_price[symbol] = ask


def set_position_leverage(exchange, symbol, leverage):
    position = exchange.fetch_positions(symbols=[symbol])[0]
    if int(position['info']['leverage']) != leverage:
        exchange.set_leverage(leverage, symbol=symbol, params={})


def get_order_params(data, symbol, position_strat, order_type):
    if order_type == 'buy':
        if data.df['buy_breakout'][-1] == 1:
            position_strat[symbol] = 1
            return {'stopLossPrice': data.df['stop_loss2'][-1], 'position_idx': 0}
        else:
            position_strat[symbol] = 0
            return {'stopLossPrice': data.df['stop_loss'][-1] * 0.99, 'takeProfitPrice': data.df['sell_price'][-1] * 1.01, 'position_idx': 0}
    else:
        if data.df['sell_breakout'][-1] == 1:
            position_strat[symbol] = 1
            return {'stopLossPrice': data.df['stop_loss_s2'][-1], 'position_idx': 0}
        else:
            position_strat[symbol] = 0
            return {'stopLossPrice': data.df['stop_loss_s'][-1] * 1.01, 'takeProfitPrice': data.df['sell_price_s'][-1] * 0.99, 'position_idx': 0}


def handle_open_orders(exchange, new_data, last_bar, entry_price):
    if new_data:
        exchange_timestamp = exchange.milliseconds()
        for symbol in SYMBOLS:
            orders = exchange.fetch_open_orders(symbol)
            if len(orders) > 0:
                timestamp = orders[0]['timestamp']
                if (exchange_timestamp - timestamp) >= TIMEFRAME_SEC * 1000:
                    ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=2)
                    high = ohlcv[-1][1]
                    low = ohlcv[-1][2]
                    price = orders[0]['price']
                    take_profit = float(orders[0]['info']['take_profit'])
                    stop_loss = float(orders[0]['info']['stop_loss'])
                    id = orders[0]['id']
                    side = orders[0]['info']['side']
                    amount = orders[0]['amount']
                    in_position = 0
                    positions = exchange.fetch_positions(symbols=[symbol])
                    if positions[0]['info']['size'] != '0':
                        if side != positions[0]['info']['side']:
                            in_position = 1
                    if side == 'Buy' and not in_position:
                        if should_cancel_order(symbol, high, entry_price, stop_loss, 'Buy'):
                            print('Cancelling order for %s' % symbol)
                            exchange.cancel_order(id, symbol=symbol)
                        else:
                            adjust_order(exchange, symbol, id, side, amount, price, stop_loss, take_profit)
                    elif not in_position:
                        if should_cancel_order(symbol, low, entry_price, stop_loss, 'Sell'):
                            print('Cancelling order for %s' % symbol)
                            exchange.cancel_order(id, symbol=symbol)
                        else:
                            adjust_order(exchange, symbol, id, side, amount, price, stop_loss, take_profit)


def should_cancel_order(symbol, price, entry_price, stop_loss, side):
    if symbol not in entry_price:
        return True
    if side == 'Buy':
        return (price - entry_price[symbol]) / (entry_price[symbol] - stop_loss) > 0.5
    return (entry_price[symbol] - price) / (stop_loss - entry_price[symbol]) > 0.5


def adjust_order(exchange, symbol, id, side, amount, price, stop_loss, take_profit):
    result = exchange.fetch_order_book(symbol, limit=2)
    if side == 'Buy':
        new_price = result['bids'][0][0]
    else:
        new_price = result['asks'][0][0]

    params = {
        'stopLossPrice': stop_loss,
        'takeProfitPrice': take_profit + new_price - price,
        'position_idx': 0
    }
    print('Editing bid price for %s' % symbol)
    exchange.cancel_order(id, symbol=symbol)
    exchange.create_order(symbol, 'limit', side, amount, new_price, params)


def plot_trade(data):
    bars = data.df
    last_bought = len(bars) - 1
    sr_buffer = 15
    ## PLOT SUPPORT AND RESISTANCE##
    end = len(bars) - 1
    j = end
    fig = go.Figure(data=[go.Candlestick(
        x=bars.index[last_bought - (200 + sr_buffer):min(j + 100, end)],
        open=bars['open'][last_bought - (200 + sr_buffer):min(j + 100, end)],
        high=bars['high'][last_bought - (200 + sr_buffer):min(j + 100, end)],
        low=bars['low'][last_bought - (200 + sr_buffer):min(j + 100, end)],
        close=bars['close'][last_bought - (200 + sr_buffer):min(j + 100, end)]
    )])

    centers_long = data.centers_plot[last_bought]
    support = data.support[last_bought]
    resistance = data.resistance[last_bought]

def plot_trade(data):
    bars = data.df
    last_bought = len(bars)-1
    sr_buffer = 15
    ## PLOT SUPPORT AND RESISTANCE##
    end = len(bars)-1
    j = end
    fig = go.Figure(data=[go.Candlestick(x=bars.index[last_bought - (200+sr_buffer):min(j+100, end)],
                                         open=bars['open'][last_bought - (200+sr_buffer):min(j+100, end)],
                                         high=bars['high'][last_bought - (200+sr_buffer):min(j+100, end)],
                                         low=bars['low'][last_bought - (200+sr_buffer):min(j+100, end)],
                                         close=bars['close'][last_bought - (200+sr_buffer):min(j+100, end)])])

    centers_long = data.centers_plot[last_bought]
    support = data.support[last_bought]
    resistance = data.resistance[last_bought]

    for center in support:
        fig.add_trace(go.Scatter(x=[bars.index[last_bought - (200+sr_buffer)],
                                    bars.index[last_bought - (200+sr_buffer)],
                                    bars.index[min(j+100, end)], bars.index[min(j+100, end)],
                                    bars.index[last_bought - (200+sr_buffer)]],
                                 y=[center[0], center[1], center[1], center[0], center[0]], fill="toself",
                                 opacity=0.25, mode="lines", fillcolor="Green", line=dict(color="Green")))
    for center in resistance:
        fig.add_trace(go.Scatter(x=[bars.index[last_bought - (200+sr_buffer)],
                                    bars.index[last_bought - (200+sr_buffer)],
                                    bars.index[min(j+100, end)], bars.index[min(j+100, end)],
                                    bars.index[last_bought - (200+sr_buffer)]],
                                 y=[center[0], center[1], center[1], center[0], center[0]], fill="toself",
                                 opacity=0.25, mode="lines", fillcolor="Red", line=dict(color="Red")))
    for center in centers_long:
        fig.add_trace(go.Scatter(x=[bars.index[last_bought - (200 + sr_buffer)],
                                    bars.index[last_bought - (200 + sr_buffer)],
                                    bars.index[min(j+100, end)], bars.index[min(j+100, end)],
                                    bars.index[last_bought - (200 + sr_buffer)]],
                                 y=[center[0], center[1], center[1], center[0], center[0]], fill="toself",
                                 opacity=0.25, mode="lines", fillcolor="Blue", line=dict(color="Blue")))

    fig.add_vline(x=bars.index[last_bought - sr_buffer], line=dict(color='Black', dash='dash'), opacity = 0.25)
    fig.add_vline(x=bars.index[last_bought], opacity = 0.25)

    sma_low = (bars.sma_high+bars.sma_low)/2
    sma_high = sma_low
    if bars['low'][last_bought] > bars['ema'][last_bought]:
        fig.add_trace(
            go.Scatter(x=bars.index[last_bought - (200+sr_buffer):last_bought],
                       y=sma_low[last_bought - (200+sr_buffer):last_bought],
                       mode='lines',
                       line=dict(color="Blue")
                       ))
    else:
        fig.add_trace(
            go.Scatter(x=bars.index[last_bought - (200+sr_buffer):last_bought],
                       y=sma_high[last_bought - (200+sr_buffer):last_bought],
                       mode='lines',
                       line=dict(color="Blue")
                       ))

    fig.add_vline(x=bars.index[int(last_bought - bars.turningpoints[last_bought])], opacity=0.25)
    fig.add_vline(x=bars.index[j-1], opacity=0.25)

    fig.add_trace(
        go.Scatter(x=bars.index[last_bought - sr_buffer:last_bought], y=bars.upperband[last_bought - sr_buffer:last_bought],
                   mode='lines',
                   line=dict(color="#2f98de")
                   ))
    fig.add_trace(
        go.Scatter(x=bars.index[last_bought - sr_buffer:last_bought], y=bars.lowerband[last_bought - sr_buffer:last_bought],
                   mode='lines',
                   line=dict(color="#2f98de")
                   ))
    fig.add_trace(
        go.Scatter(x=bars.index[last_bought - (200+sr_buffer):j], y=bars.ema[last_bought - (200+sr_buffer):min(j+sr_buffer, end)],
                   mode='lines',
                   line=dict(color="#ff9933")
                   ))

    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.write_html(f"images/fig{j}.html")

