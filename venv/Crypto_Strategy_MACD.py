import numpy as np
import talib as ta
import pandas as pd
import datetime as dt

# Constants
ATR_LENGTH = 14
ATR_MULTIPLIER = 2
SL_RATIO = 1.20 / 1.25
TP_RATIO = 1.25
EMA_LENGTH = 200
MACD_FAST_PERIOD = 9
MACD_SLOW_PERIOD = 22
MACD_SIGNAL_PERIOD = 6
ADX_LENGTH = 100
ADX_SMOOTH = 50
DATA_WINDOW = 800

def tr(data):
    """Calculate True Range (TR)."""
    data['previous_close'] = data['close'].shift(1)
    data['high-low'] = abs(data['high'] - data['low'])
    data['high-pc'] = abs(data['high'] - data['previous_close'])
    data['low-pc'] = abs(data['low'] - data['previous_close'])
    tr = data[['high-low', 'high-pc', 'low-pc']].max(axis=1)
    return tr

def atr(data, period):
    """Calculate Average True Range (ATR)."""
    data['tr'] = tr(data)
    atr = data['tr'].rolling(period).mean()
    return atr

def adx_ma(data, period=ADX_LENGTH, smooth=ADX_SMOOTH, limit=18, stream=0):
    """Calculate ADX and Moving Averages."""
    end = len(data)
    if 'up' not in data:
        initialize_adx_columns(data)
    calculate_tr(data, period, end, stream)
    calculate_dm(data, period, end, stream)
    calculate_adx(data, period, smooth, end, stream)
    return data

def initialize_adx_columns(data):
    """Initialize ADX related columns."""
    data['tr'] = np.nan
    data['adx'] = 0
    data['adx_coarse'] = 0
    data['up'] = 0
    data['down'] = 0
    data['plus'] = 0
    data['minus'] = 0

def calculate_tr(data, period, end, stream):
    """Calculate True Range (TR)."""
    if stream:
        data.iloc[-1, data.columns.get_loc('tr')] = ta.stream_TRANGE(data['high'], data['low'], data['close'])
    else:
        data['tr'] = ta.TRANGE(data['high'], data['low'], data['close'])

def calculate_dm(data, period, end, stream):
    """Calculate Directional Movement (DM)."""
    start = 0 if not stream else len(data) - 1
    data.iloc[start:end, data.columns.get_loc('up')] = data.high.diff()
    data.iloc[start:end, data.columns.get_loc('down')] = data.low.diff() * -1
    data.iloc[start:end, data.columns.get_loc('plus')] = np.where((data.up > data.down) & (data.up > 0), data.up, 0)
    data.iloc[start:end, data.columns.get_loc('minus')] = np.where((data.down > data.up) & (data.down > 0), data.down, 0)

def calculate_adx(data, period, smooth, end, stream):
    """Calculate Average Directional Index (ADX)."""
    start = 0 if not stream else len(data) - 1
    trur = data.tr.ewm(alpha=1 / period).mean()
    plus = 100 * data.plus.ewm(alpha=1 / period).mean() / trur
    minus = 100 * data.minus.ewm(alpha=1 / period).mean() / trur
    sum = plus + minus
    data.iloc[start:end, data.columns.get_loc('adx_coarse')] = np.abs(plus - minus) / np.where(sum == 0, 1, sum)
    adxsmooth = 100 * data.adx_coarse.ewm(alpha=1 / smooth).mean()
    data.iloc[start:end, data.columns.get_loc('adx')] = adxsmooth[start:end]

def check_hl(data_back, data_forward, hl):
    """Check if a high or low is valid."""
    ref = data_back[-1]
    if hl.lower() == 'high':
        return all(ref >= x for x in data_back[:-1]) and all(ref > x for x in data_forward)
    if hl.lower() == 'low':
        return all(ref <= x for x in data_back[:-1]) and all(ref < x for x in data_forward)
    return False

def pivot(osc, LBL, LBR, highlow, stream=0, pivots=[], left=[], right=[]):
    """Identify pivot points."""
    start = 0 if not stream else len(osc) - 1
    for i in range(start, len(osc)):
        if stream:
            pivots[-1] = 0.0 if np.isnan(pivots[0]) else pivots[-1]
        else:
            pivots.append(0.0)
        if i < LBL + 1:
            left.append(osc[i])
        if i > LBL:
            right.append(osc[i])
        if i > LBL + LBR:
            left.append(right.pop(0))
            left.pop(0)
            if check_hl(left, right, highlow):
                pivots[i] = osc[i - LBR]
    return pivots, left, right

def pivot_price(data, left_bars, right_bars, stream=0):
    """Calculate pivot prices."""
    if 'pivots_high' not in data.df.columns:
        initialize_pivot_columns(data)
    update_pivot_data(data, left_bars, right_bars, stream)
    update_pivot_prices(data, left_bars, right_bars, stream)
    return data

def initialize_pivot_columns(data):
    """Initialize pivot related columns."""
    data.df['pivots_high'] = np.nan
    data.df['pivots_low'] = np.nan
    data.df['highprice'] = np.nan
    data.df['lowprice'] = np.nan
    data.misc['pivots_left_h'] = []
    data.misc['pivots_right_h'] = []
    data.misc['pivots_left_l'] = []
    data.misc['pivots_right_l'] = []

def update_pivot_data(data, left_bars, right_bars, stream):
    """Update pivot data."""
    pivots_high, l, r = pivot(data.df['high'], left_bars, right_bars, 'high', stream,
                              data.df.pivots_high.copy(), data.misc['pivots_left_h'], data.misc['pivots_right_h'])
    data.df['pivots_high'] = pivots_high
    data.misc['pivots_left_h'] = l
    data.misc['pivots_right_h'] = r
    pivots_low, l, r = pivot(data.df['low'], left_bars, right_bars, 'low', stream,
                             data.df.pivots_low.copy(), data.misc['pivots_left_l'], data.misc['pivots_right_l'])
    data.df['pivots_low'] = pivots_low
    data.misc['pivots_left_l'] = l
    data.misc['pivots_right_l'] = r

def update_pivot_prices(data, left_bars, right_bars, stream):
    """Update pivot prices."""
    start = 0 if not stream else len(data.df) - 1
    for i in range(max(start, 0), len(data.df)):
        if data.df['highprice'][i-1] != data.df['pivots_high'][i] and data.df['pivots_high'][i] != 0:
            data.df.iloc[i, data.df.columns.get_loc('highprice')] = data.df['pivots_high'][i]
        else:
            data.df.iloc[i, data.df.columns.get_loc('highprice')] = data.df['highprice'][i-1]
        if data.df['lowprice'][i-1] != data.df['pivots_low'][i] and data.df['pivots_low'][i] != 0:
            data.df.iloc[i, data.df.columns.get_loc('lowprice')] = data.df['pivots_low'][i]
        else:
            data.df.iloc[i, data.df.columns.get_loc('lowprice')] = data.df['lowprice'][i-1]

def crossover_series(x: pd.Series, y: pd.Series, stream=0) -> pd.Series:
    """Identify crossover points in series."""
    shift_value = 1
    out = (x > y) & (x.shift(shift_value) < y.shift(shift_value))
    return out[-1] if stream else out

def crossunder_series(x: pd.Series, y: pd.Series, stream=0) -> pd.Series:
    """Identify crossunder points in series."""
    shift_value = 1
    out = (x < y) & (x.shift(shift_value) > y.shift(shift_value))
    return out[-1] if stream else out

def strategy(data, ATRlength=ATR_LENGTH, ATRmultiplier=ATR_MULTIPLIER, SLratio=SL_RATIO, TPratio=TP_RATIO,
             emalength=EMA_LENGTH, MACDfastperiod=MACD_FAST_PERIOD, MACDslowperiod=MACD_SLOW_PERIOD, MACDlength=MACD_SIGNAL_PERIOD,
             ADXlength=ADX_LENGTH, ADXsmooth=ADX_SMOOTH, data_window=DATA_WINDOW, stream=0):
    """Trading strategy implementation."""
    df = data.df
    length = len(df)
    initialize_strategy_columns(df, data_window)

    if stream:
        data = pivot_price(data, 15, 15, stream=1)
        data = pivot_price2(data, 6, 3, stream=1)
        df = adx_ma(df, ADXlength, ADXsmooth, stream=1)
        ATRvalue = ta.stream_ATR(df['high'], df['low'], df['close'], timeperiod=ATRlength)
        start = length - 1
        ema = ta.EMA(df['close'][max(0, length-data_window):length], timeperiod=emalength)[-1]
        macd, macdsignal, macdhist = ta.MACD(df['close'], fastperiod=MACDfastperiod, slowperiod=MACDslowperiod, signalperiod=MACDlength)
        macd, macdsignal, macdhist = macd[-1], macdsignal[-1], macdhist[-1]
    else:
        data = pivot_price(data, 15, 15)
        data = pivot_price2(data, 6, 3)
        start = 0
        df = adx_ma(df, ADXlength, ADXsmooth, stream=0)
        ATRvalue = ta.ATR(df['high'], df['low'], df['close'], timeperiod=ATRlength)
        ema = ta.EMA(df['close'], timeperiod=emalength)
        macd, macdsignal, macdhist = ta.MACD(df['close'], fastperiod=MACDfastperiod, slowperiod=MACDslowperiod, signalperiod=MACDlength)

    calculate_stop_loss_take_profit(df, ATRvalue, ATRmultiplier, ema, start, length)
    add_indicators_to_dataframe(df, ema, macdhist, start, length)
    apply_trading_rules(df, macd, macdsignal, start, length, stream)

    return data

def initialize_strategy_columns(df, data_window):
    """Initialize columns for strategy."""
    if 'ema' not in df:
        df['atr'] = np.nan
        df['upperband'] = np.nan
        df['lowerband'] = np.nan
        df['in_uptrend'] = 1
        df['ema'] = np.nan
        df['shortLossValue'] = np.nan
        df['longLossValue'] = np.nan
        df['macdhist'] = np.nan

def calculate_stop_loss_take_profit(df, ATRvalue, ATRmultiplier, ema, start, length):
    """Calculate stop loss and take profit values."""
    df['shortLossValue'] = df['high'][start:length] + ATRmultiplier * ATRvalue
    df['longLossValue'] = df['low'][start:length] - ATRmultiplier * ATRvalue
    tp_offset_long = (df['close'][start:length] - np.minimum(ema, df['longLossValue'])) * TPratio
    df['stop_loss'] = (df['close'][start:length] - SLratio * tp_offset_long) * df['buy']
    df['sell_price'] = df['close'][start:length] + tp_offset_long
    tp_offset_short = (df['close'][start:length] - np.maximum(ema, df['shortLossValue'])) * TPratio
    df['stop_loss_s'] = (df['close'][start:length] - SLratio * tp_offset_short) * df['short']
    df['sell_price_s'] = df['close'][start:length] + tp_offset_short

def add_indicators_to_dataframe(df, ema, macdhist, start, length):
    """Add indicator values to the dataframe."""
    df['ema'][start:length] = ema
    df['macdhist'][start:length] = macdhist

def apply_trading_rules(df, macd, macdsignal, start, length, stream):
    """Apply trading rules to generate buy and sell signals."""
    near_resistance = (np.abs(df['close'][start:length] - df['highprice'][start:length]) < np.abs(df['close'][start:length] - df['lowprice'][start:length])) \
        & ((df['highprice'][start:length] - df['highprice2'][start:length]) < 0.3 * (df['highprice'][start:length] - df['lowprice'][start:length])) \
        & ((df['highprice'][start:length] - df['highprice2'][start:length]) != 0)

    near_support = (np.abs(df['close'][start:length] - df['lowprice'][start:length]) < np.abs(df['close'][start:length] - df['highprice'][start:length])) \
        & ((df['lowprice2'][start:length] - df['lowprice'][start:length]) < 0.3 * (df['highprice'][start:length] - df['lowprice'][start:length])) \
        & ((df['lowprice'][start:length] - df['lowprice2'][start:length]) != 0)

    buy_signal = (crossover_series(df['macdhist'][max(start-1, 0):length], 0 * df['macdhist'][max(start-1, 0):length], stream)) \
        & (df['low'][start:length] > df['ema'][start:length]) & (macd < 0) & (macdsignal < 0) & (df['adx'][start:length] > 15) \
        & (df['ema'][start:length] > df['ema'].shift(1)[start:length]) & near_support & (df['ema'][start:length] < df['longLossValue'][start:length])

    sell_signal = (crossunder_series(df['macdhist'][max(start-1, 0):length], 0 * df['macdhist'][max(start-1, 0):length], stream)) \
        & (df['high'][start:length] < df['ema'][start:length]) & (macd > 0) & (macdsignal > 0) & (df['adx'][start:length] > 15) \
        & (df['ema'][start:length] < df['ema'].shift(1)[start:length]) & near_resistance & (df['ema'][start:length] > df['shortLossValue'][start:length])

    df['buy'][start:length] = buy_signal.astype(int)
    df['short'][start:length] = sell_signal.astype(int)