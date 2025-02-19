import os
import numpy as np
import talib as ta
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

# Set environment variables to limit the number of threads used by numpy
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


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


def initialize_adx_columns(data):
    """Initialize ADX related columns."""
    data['tr'] = np.nan
    data['adx'] = 0
    data['adx_coarse'] = 0
    data['up'] = 0
    data['down'] = 0
    data['plus'] = 0
    data['minus'] = 0


def adx_ma(data, period=14, smooth=14, limit=18, stream=0):
    """Calculate ADX and Moving Averages."""
    end = len(data)

    if 'up' not in data:
        initialize_adx_columns(data)

    if stream:
        start = len(data) - 1
        data.iloc[-1, data.columns.get_loc('tr')] = ta.stream_TRANGE(data['high'], data['low'], data['close'])
        if end < 2:
            return data
        elif end < period + 3:
            update_adx_initial(data)
        else:
            update_adx_values(data, end)
    else:
        start = 0
        data['tr'] = ta.TRANGE(data['high'], data['low'], data['close'])
        data.iloc[start:end, data.columns.get_loc('up')] = data.high.diff()
        data.iloc[start:end, data.columns.get_loc('down')] = data.low.diff() * -1

    calculate_adx(data, period, smooth, start, end, stream)
    return data


def update_adx_initial(data):
    """Update initial ADX values."""
    data.iloc[-1, data.columns.get_loc('up')] = data.high[-1] - data.high[-2]
    data.iloc[-1, data.columns.get_loc('down')] = data.low[-2] - data.low[-1]
    data.iloc[-1, data.columns.get_loc('adx')] = 0
    data.iloc[-1, data.columns.get_loc('adx_coarse')] = 0
    data.iloc[-1, data.columns.get_loc('plus')] = 0
    data.iloc[-1, data.columns.get_loc('minus')] = 0


def update_adx_values(data, end):
    """Update ADX values."""
    data.iloc[-1, data.columns.get_loc('up')] = data.high[-1] - data.high[-2]
    data.iloc[-1, data.columns.get_loc('down')] = data.low[-2] - data.low[-1]


def calculate_adx(data, period, smooth, start, end, stream):
    """Calculate Average Directional Index (ADX)."""
    data.iloc[start:end, data.columns.get_loc('plus')] = np.where(
        ((data.up[start:end] > data.down[start:end]) & (data.up[start:end] > 0)), data.up[start:end], 0)
    data.iloc[start:end, data.columns.get_loc('minus')] = np.where(
        ((data.down[start:end] > data.up[start:end]) & (data.down[start:end] > 0)), data.down[start:end], 0)

    trur = data.tr.ewm(alpha=1 / period).mean()
    plus = 100 * data.plus.ewm(alpha=1 / period).mean() / trur
    minus = 100 * data.minus.ewm(alpha=1 / period).mean() / trur

    if stream:
        if end < period + 3:
            plus = 0
            minus = 0
    else:
        plus = np.r_[np.zeros(period + 2), plus[(period + 2):]]
        minus = np.r_[np.zeros(period + 2), minus[(period + 2):]]

    sum_values = plus + minus
    data.iloc[start:end, data.columns.get_loc('adx_coarse')] = np.abs(plus - minus) / np.where(sum_values == 0, 1, sum_values)
    adxsmooth = 100 * data.adx_coarse.ewm(alpha=1 / smooth).mean()
    data.iloc[start:end, data.columns.get_loc('adx')] = adxsmooth[start:end]

    if stream:
        if end < period + 3:
            data.iloc[-1, data.columns.get_loc('adx')] = 0
            data.iloc[-1, data.columns.get_loc('adx_coarse')] = 0
    else:
        data.iloc[start:end, data.columns.get_loc('adx')] = np.r_[np.zeros(smooth + 2), data.adx[(smooth + 2):]]


def check_hl(data_back, data_forward, hl):
    """Check if a high or low is valid."""
    ref = data_back[-1]
    if hl.lower() == 'high':
        return all(ref >= x for x in data_back[:-1]) and all(ref > x for x in data_forward)
    if hl.lower() == 'low':
        return all(ref <= x for x in data_back[:-1]) and all(ref < x for x in data_forward)
    return False


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


def find_support_resistance(data, max_clusters=5, k_add=0, override=0):
    """Identify support and resistance levels using KMeans clustering."""
    mh = pd.Series(data['high']).values
    ml = pd.Series(data['low']).values

    if override != 0:
        return override_support_resistance(mh, ml, override)

    return optimal_support_resistance(mh, ml, max_clusters, k_add)


def override_support_resistance(mh, ml, override):
    """Override support and resistance levels using provided cluster counts."""
    kmeans_h = KMeans(n_clusters=override[0], random_state=0, n_init=5).fit(mh.reshape(-1, 1))
    db_score_h = davies_bouldin_score(mh.reshape(-1, 1), kmeans_h.labels_)
    kmeans_l = KMeans(n_clusters=override[1], random_state=0, n_init=5).fit(ml.reshape(-1, 1))
    db_score_l = davies_bouldin_score(ml.reshape(-1, 1), kmeans_l.labels_)
    db_max = max(db_score_h, db_score_l)
    return kmeans_h, kmeans_l, db_max


def optimal_support_resistance(mh, ml, max_clusters, k_add):
    """Identify optimal support and resistance levels using KMeans clustering."""
    inertia_h, k_model_h, db_h = calculate_kmeans(mh, max_clusters)
    elbow_h, db_score_h = determine_elbow_point(inertia_h, db_h, k_add)

    inertia_l, k_model_l, db_l = calculate_kmeans(ml, max_clusters)
    elbow_l, db_score_l = determine_elbow_point(inertia_l, db_l, k_add)

    elbow, db_max = (elbow_h, db_score_h) if db_score_h > db_score_l else (elbow_l, db_score_l)

    return k_model_h[elbow], k_model_l[elbow], db_max


def calculate_kmeans(data, max_clusters):
    """Calculate KMeans clustering for a range of cluster counts."""
    inertia = []
    k_model = {}
    db = []

    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=5).fit(data.reshape(-1, 1))
        k_model[k] = kmeans
        inertia.append(kmeans.inertia_)
        if k > 1:
            labels = kmeans.labels_
            db_score = davies_bouldin_score(data.reshape(-1, 1), labels)
            db.append(db_score)

    return inertia, k_model, db


def determine_elbow_point(inertia, db, k_add):
    """Determine the elbow point in the inertia plot."""
    for m in range(1, len(inertia)):
        delta = (inertia[m - 1] - inertia[m]) / inertia[m - 1]
        if delta < 0.5:
            break
    elbow = max(k_add, np.argmax(db) + 2)
    db_score = db[elbow - 2]
    return elbow, db_score


def gen_stochf(bars, tp):
    """Generate Stochastic Fast indicator."""
    bars['fastk'], fastd = ta.STOCHF(bars['high'], bars['low'], bars['close'], fastk_period=tp)
    return bars


def turningpoints(lst):
    """Identify turning points in a list."""
    dx = np.diff(lst)
    return np.sum((dx[1:] * dx[:-1]) < 0)


def sort_sr(centers_high, centers_low, atr):
    """Sort and merge support and resistance levels."""
    centers = np.sort(np.concatenate((centers_high, centers_low)).flatten())
    centers_new = []

    while len(centers) > 0:
        if len(centers) == 1:
            centers_new.append([centers[0], centers[0]])
            centers = np.delete(centers, 0)
            break

        dk_min, n = find_closest_centers(centers)
        if dk_min > atr * 4:
            centers_new.append([centers[n], centers[n]])
            centers_new.append([centers[n - 1], centers[n - 1]])
            centers = np.delete(centers, [n, n - 1])
        else:
            centers_new.append([centers[n - 1], centers[n]])
            centers = np.delete(centers, [n, n - 1])

    centers_new = sorted(centers_new, key=lambda x: x[0])
    return centers_new


def find_closest_centers(centers):
    """Find the closest centers in the list."""
    dk_min = np.inf
    n = 0
    for k in range(1, len(centers)):
        dk = centers[k] - centers[k - 1]
        if dk < dk_min:
            dk_min = dk
            n = k
    return dk_min, n


def update_touch(touch_old, touch_new, bounce_up, bounce_down, through_up, through_down):
    """Update touch points."""
    if touch_old[-1] != touch_new:
        if np.abs(touch_old[-1] - touch_new) == 1:
            touch_old.append(touch_new)
        elif touch_new > touch_old[-1]:
            touch_old.extend(range(touch_old[-1] + 1, touch_new + 1))
        elif touch_new < touch_old[-1]:
            touch_old.extend(range(touch_old[-1] - 1, touch_new - 1, -1))
    if len(touch_old) >= 3:
        n_end = len(touch_old) - 2
        for n in range(1, n_end + 1):
            if touch_old[0] < touch_old[1] < touch_old[2]:
                through_up[touch_old[1]] += 1
            elif touch_old[2] < touch_old[1] < touch_old[0]:
                through_down[touch_old[1]] += 1
            elif (touch_old[1] > touch_old[0]) and (touch_old[1] > touch_old[2]):
                bounce_down[touch_old[1]] += 1
            else:
                bounce_up[touch_old[1]] += 1
        del touch_old[0]
    return touch_old, bounce_up, bounce_down, through_up, through_down


def decide_sr(centers, high, low, atr, close):
    """Decide support and resistance levels."""
    bounce_up = np.zeros(len(centers))
    bounce_down = np.zeros(len(centers))
    through_up = np.zeros(len(centers))
    through_down = np.zeros(len(centers))

    high = (high + low) / 2
    low = (high + low) / 2

    atr_av = np.median(atr)
    for center in centers:
        if center[1] - center[0] < 0.4 * atr_av:
            center_mid = np.mean(center)
            center[0] = center_mid - 0.2 * atr_av
            center[1] = center_mid + 0.2 * atr_av

    touch_old = []
    touch_new = []
    for j in range(0, len(high)):
        touch_new = find_touch_points(centers, high[j], low[j])
        touch_new = list(set(touch_new))
        if not touch_old:
            touch_old = touch_new
        elif len(touch_new) == 1 and touch_old[-1] != touch_new[0]:
            touch_old, bounce_up, bounce_down, through_up, through_down = update_touch(
                touch_old, touch_new[0], bounce_up, bounce_down, through_up, through_down
            )
        elif len(touch_new) == 2:
            touch_old, bounce_up, bounce_down, through_up, through_down = handle_two_touch_points(
                touch_old, touch_new, bounce_up, bounce_down, through_up, through_down
            )
        elif len(touch_new) > 2:
            touch_old, bounce_up, bounce_down, through_up, through_down = handle_multiple_touch_points(
                touch_old, touch_new, close[j], centers, bounce_up, bounce_down, through_up, through_down
            )
        touch_new = []

    support, resistance = classify_support_resistance(centers, bounce_up, through_down, bounce_down, through_up)
    return support, resistance, np.where(bounce_up > 0)[0], np.where(bounce_down > 0)[0]


def find_touch_points(centers, high, low):
    """Find touch points for given high and low values."""
    touch_new = []
    for k in range(len(centers)):
        if centers[k][0] <= high <= centers[k][1] or centers[k][0] <= low <= centers[k][1]:
            touch_new.append(k)
        elif high > centers[k][1] and low < centers[k][0]:
            touch_new.append(k)
    return touch_new


def handle_two_touch_points(touch_old, touch_new, bounce_up, bounce_down, through_up, through_down):
    """Handle cases with two touch points."""
    if touch_old[-1] in touch_new:
        touch_new.remove(touch_old[-1])
        touch_old, bounce_up, bounce_down, through_up, through_down = update_touch(
            touch_old, touch_new[0], bounce_up, bounce_down, through_up, through_down
        )
    else:
        if touch_new[0] > touch_old[-1]:
            touch_new.sort()
        else:
            touch_new.sort(reverse=True)
        touch_old, bounce_up, bounce_down, through_up, through_down = update_touch(
            touch_old, touch_new[0], bounce_up, bounce_down, through_up, through_down
        )
        touch_old, bounce_up, bounce_down, through_up, through_down = update_touch(
            touch_old, touch_new[1], bounce_up, bounce_down, through_up, through_down
        )
    return touch_old, bounce_up, bounce_down, through_up, through_down


def handle_multiple_touch_points(touch_old, touch_new, close, centers, bounce_up, bounce_down, through_up, through_down):
    """Handle cases with multiple touch points."""
    if close < centers[touch_old[-1]][0]:
        touch_new.sort(reverse=True)
    elif close > centers[touch_old[-1]][1]:
        touch_new.sort()
    elif touch_old[-1] > touch_old[-2]:
        touch_new.sort()
    else:
        touch_new.sort(reverse=True)
    if touch_old[-1] in touch_new:
        touch_new.remove(touch_old[-1])
    for k in touch_new:
        touch_old, bounce_up, bounce_down, through_up, through_down = update_touch(
            touch_old, k, bounce_up, bounce_down, through_up, through_down
        )
    return touch_old, bounce_up, bounce_down, through_up, through_down


def classify_support_resistance(centers, bounce_up, through_down, bounce_down, through_up):
    """Classify support and resistance levels."""
    support = []
    resistance = []
    for k in range(len(centers)):
        if bounce_up[k] > 0 and through_down[k] == 0:
            support.append(centers[k])
        if bounce_down[k] > 0 and through_up[k] == 0:
            resistance.append(centers[k])
    return support, resistance


def strategy(data, ATRlength=14, ATRmultiplier=2, SLratio=1.20 / 1.5, TPratio=1.25, emalength=200,
             MACDfastperiod=9, MACDslowperiod=22, MACDlength=6, ADXlength=100, ADXsmooth=50, data_window=800,
             stream=0, stochf_period=14, SRbuffer=15, istart=0, tl=0):
    """Trading strategy implementation."""
    df = data.df
    length = len(df)

    if 'ema' not in df:
        initialize_strategy_columns(df)

    if stream:
        start = len(df) - 1
        ATRvalue, ema, ema2, sma_low, sma_high, macd, macdsignal, macdhist, fastk = stream_values(df, length,
                                                                                                  data_window,
                                                                                                  ATRlength, emalength,
                                                                                                  MACDfastperiod,
                                                                                                  MACDslowperiod,
                                                                                                  MACDlength,
                                                                                                  stochf_period)
    else:
        start = 0
        ATRvalue, ema, ema2, sma_low, sma_high, macd, macdsignal, macdhist, fastk = batch_values(df, ATRlength,
                                                                                                 emalength,
                                                                                                 MACDfastperiod,
                                                                                                 MACDslowperiod,
                                                                                                 MACDlength,
                                                                                                 stochf_period)

    shortLossValue, longLossValue = calculate_stop_loss(df, ATRvalue, ATRmultiplier, start, length)
    add_indicator_values(df, ATRvalue, ema, ema2, sma_low, sma_high, shortLossValue, longLossValue, macdhist, fastk,
                         start, length)

    add = 0
    if not stream:
        start = istart
        start1 = istart + 1
        if istart == 0:
            add = 751 + SRbuffer
    else:
        start1 = start

    for i in range(max([751 + SRbuffer, start]), length):
        if (i + add + tl - max([751 + SRbuffer, start])) % 100 == 0:
            print((i + add + tl - max([751 + SRbuffer, start])))
        process_strategy(data, df, i, SRbuffer, stream)

    finalize_strategy(df, start, length, stream)
    return data


def initialize_strategy_columns(df):
    """Initialize columns for strategy."""
    df['atr'] = np.nan
    df['upperband'] = np.nan
    df['lowerband'] = np.nan
    df['in_uptrend'] = 1
    df['ema'] = np.nan
    df['ema2'] = np.nan
    df['sma_low'] = np.nan
    df['sma_high'] = np.nan
    df['shortLossValue'] = np.nan
    df['longLossValue'] = np.nan
    df['macdhist'] = np.nan
    df['in_sr'] = np.nan
    df['in_sr_bull'] = np.nan
    df['in_sr_bull2'] = np.nan
    df['in_sr_bear'] = np.nan
    df['in_sr_bear2'] = np.nan
    df['sl_bull'] = np.nan
    df['sl_bear'] = np.nan
    df['inertia_long'] = np.nan
    df['inertia_short'] = np.nan
    df['fastk'] = np.nan
    df['peak'] = np.nan
    df['trough'] = np.nan
    df['turningpoints'] = np.nan

    data.misc['override_long'] = 0
    data.misc['trigger_override_long'] = 0
    data.misc['inertia_long'] = []


def stream_values(df, length, data_window, ATRlength, emalength, MACDfastperiod, MACDslowperiod, MACDlength,
                  stochf_period):
    """Calculate values for streaming mode."""
    ATRvalue = ta.stream_ATR(df['high'], df['low'], df['close'], timeperiod=ATRlength)
    ema = ta.EMA(df['close'][max(0, length - data_window):length], timeperiod=emalength)[-1]
    ema2 = ta.EMA(df['close'][max(0, length - data_window):length], timeperiod=emalength / 2)[-1]
    sma_low = ta.stream_SMA(df['low'], timeperiod=3)
    sma_high = ta.stream_SMA(df['high'], timeperiod=3)
    macd, macdsignal, macdhist = ta.MACD(df['close'], fastperiod=MACDfastperiod, slowperiod=MACDslowperiod,
                                         signalperiod=MACDlength)
    macd, macdsignal, macdhist = macd[-1], macdsignal[-1], macdhist[-1]
    fastk, fastd = ta.stream_STOCHF(df['high'], df['low'], df['close'], fastk_period=stochf_period)
    return ATRvalue, ema, ema2, sma_low, sma_high, macd, macdsignal, macdhist, fastk


def batch_values(df, ATRlength, emalength, MACDfastperiod, MACDslowperiod, MACDlength, stochf_period):
    """Calculate values for batch mode."""
    ATRvalue = ta.ATR(df['high'], df['low'], df['close'], timeperiod=ATRlength)
    ema = ta.EMA(df['close'], timeperiod=emalength)
    ema2 = ta.EMA(df['close'], timeperiod=emalength / 2)
    sma_low = ta.SMA(df['low'], timeperiod=3)
    sma_high = ta.SMA(df['high'], timeperiod=3)
    macd, macdsignal, macdhist = ta.MACD(df['close'], fastperiod=MACDfastperiod, slowperiod=MACDslowperiod,
                                         signalperiod=MACDlength)
    fastk, fastd = ta.STOCHF(df['high'], df['low'], df['close'], fastk_period=stochf_period)
    return ATRvalue, ema, ema2, sma_low, sma_high, macd, macdsignal, macdhist, fastk


def calculate_stop_loss(df, ATRvalue, ATRmultiplier, start, length):
    """Calculate stop loss values."""
    shortLossValue = df['high'][start:length] + ATRmultiplier * ATRvalue * 2.0
    longLossValue = df['low'][start:length] - ATRmultiplier * ATRvalue * 2.0
    return shortLossValue, longLossValue


def add_indicator_values(df, ATRvalue, ema, ema2, sma_low, sma_high, shortLossValue, longLossValue, macdhist, fastk,
                         start, length):
    """Add indicator values to the dataframe."""
    df.iloc[start:length, df.columns.get_loc('atr')] = ATRvalue
    df.iloc[start:length, df.columns.get_loc('ema')] = ema
    df.iloc[start:length, df.columns.get_loc('ema2')] = ema2
    df.iloc[start:length, df.columns.get_loc('sma_low')] = sma_low
    df.iloc[start:length, df.columns.get_loc('sma_high')] = sma_high
    df.iloc[start:length, df.columns.get_loc('shortLossValue')] = shortLossValue
    df.iloc[start:length, df.columns.get_loc('longLossValue')] = longLossValue
    df.iloc[start:length, df.columns.get_loc('macdhist')] = macdhist
    df.iloc[start:length, df.columns.get_loc('fastk')] = fastk
    df.iloc[start:length, df.columns.get_loc('lowerband')] = df['close'][start:length] - 1 * ATRvalue
    df.iloc[start:length, df.columns.get_loc('upperband')] = df['close'][start:length] + 1 * ATRvalue


def process_strategy(data, df, i, SRbuffer, stream):
    """Process the trading strategy."""
    override_long = data.misc['override_long']
    trigger_override_long = data.misc['trigger_override_long']
    inertia_long = data.misc['inertia_long']
    if i == max([751 + SRbuffer, start]):
        data.misc['override_long'] = 0
    elif trigger_override_long > 0:
        data.misc['override_long'] = 0
        data.misc['trigger_override_long'] = 0
    elif len(inertia_long) >= 25:
        data.misc['override_long'] = 0
    else:
        data.misc['override_long'] = data.misc['centers_long']

    if override_long == 0:
        data.misc['inertia_long'] = []
    high_clusters, low_clusters, inertia = find_support_resistance(df[(i - (200 + SRbuffer)):(i - SRbuffer)], k_add=3,
                                                                   override=data.misc['override_long'])
    df.iloc[i, df.columns.get_loc('inertia_long')] = inertia
    low_centers = low_clusters.cluster_centers_
    low_centers_long = np.sort(low_centers, axis=0)
    high_centers = high_clusters.cluster_centers_
    high_centers_long = np.sort(high_centers, axis=0)
    combined_long = sort_sr(high_centers, low_centers, df['atr'][i])
    support, resistance, support_indices, resistance_indices = decide_sr(combined_long,
                                                                         df.high[(i - (200 + SRbuffer)):(i - SRbuffer)],
                                                                         df.low[(i - (200 + SRbuffer)):(i - SRbuffer)],
                                                                         df.atr[(i - (200 + SRbuffer)):(i - SRbuffer)],
                                                                         df.close[
                                                                         (i - (200 + SRbuffer)):(i - SRbuffer)])
    data.support[i] = support
    data.resistance[i] = resistance
    data.misc['centers_long'] = [len(high_centers_long), len(low_centers_long)]
    data.misc['inertia_long'].append(df['inertia_long'][i])

    if data.misc['override_long'] != 0:
        inertia_ratio_long = (max(data.misc['inertia_long']) - min(data.misc['inertia_long'])) / min(
            data.misc['inertia_long'])
        if inertia_ratio_long > 0.05:
            trigger_override_long = 1
            data.misc['trigger_override_long'] = 1

    data.sr_long[i] = combined_long
    combined_long_bull = support
    combined_long_bear = resistance
    low = df['low'][i]
    high = df['high'][i]
    in_sr_bull, in_sr_bull2, in_sr_bear, in_sr_bear2, sl_bull, sl_bear = calculate_sr_levels(combined_long,
                                                                                             combined_long_bull,
                                                                                             combined_long_bear, low,
                                                                                             high, support_indices,
                                                                                             resistance_indices)

    df.iloc[i, df.columns.get_loc('in_sr_bull')] = in_sr_bull
    df.iloc[i, df.columns.get_loc('in_sr_bull2')] = in_sr_bull2
    df.iloc[i, df.columns.get_loc('in_sr_bear')] = in_sr_bear
    df.iloc[i, df.columns.get_loc('in_sr_bear2')] = in_sr_bear2
    df.iloc[i, df.columns.get_loc('sl_bull')] = sl_bull
    df.iloc[i, df.columns.get_loc('sl_bear')] = sl_bear


def calculate_sr_levels(combined_long, combined_long_bull, combined_long_bear, low, high, support_indices,
                        resistance_indices):
    """Calculate support and resistance levels."""
    in_sr_bull = 0
    in_sr_bull2 = 0
    in_sr_bear = 0
    in_sr_bear2 = 0
    sl_bull = 0
    sl_bear = 0

    for k in range(len(combined_long_bull)):
        if (low <= combined_long_bull[k][0] <= high) or (low <= combined_long_bull[k][1] <= high) or (
        (low > combined_long_bull[k][0] and (high < combined_long_bull[k][1]))):
            ks = len(combined_long) - 1
            for ks in resistance_indices:
                if ks > support_indices[k]:
                    break
            in_sr_bull = combined_long[ks][1]
            in_sr_bull2 = combined_long[k][0]
            if sl_bull == 0:
                sl_bull = combined_long[max(support_indices[k] - 1, 0)][0]
                if support_indices[k] == 0:
                    sl_bull = 0
                    in_sr_bull = 0

    for k in range(len(combined_long_bear)):
        if (low <= combined_long_bear[k][0] <= high) or (low <= combined_long_bear[k][1] <= high) or (
                (low > combined_long_bear[k][0]) and (high < combined_long_bear[k][1])):
            ks = 0
            for ks in reversed(support_indices):
                if ks < resistance_indices[k]:
                    break
            in_sr_bear = combined_long[ks][0]
            in_sr_bear2 = combined_long[k][1]
            if sl_bear == 0:
                sl_bear = combined_long[min(resistance_indices[k] + 1, len(combined_long) - 1)][1]
                if resistance_indices[k] == len(combined_long) - 1:
                    sl_bear = 0
                    in_sr_bear = 0

    return in_sr_bull, in_sr_bull2, in_sr_bear, in_sr_bear2, sl_bull, sl_bear


def finalize_strategy(df, start, length, stream):
    """Finalize the strategy by adding buy and sell signals."""
    near_sr = df['in_sr'][start:length] == 1

    trough = (df['fastk'][start:length] > 20) & (df['fastk'].shift(1)[start:length] <= 20)
    peak = (df['fastk'][start:length] < 80) & (df['fastk'].shift(1)[start:length] >= 80)
    df.iloc[start:length, df.columns.get_loc('trough')] = trough
    df.iloc[start:length, df.columns.get_loc('peak')] = peak

    if not stream:
        macd = macd[start:length]
        macdsignal = macdsignal[start:length]

    buy_signal_first = (crossover_series(df['macdhist'][max(start1 - 1, 0):length],
                                         0 * df['macdhist'][max(start1 - 1, 0):length], stream)) & \
                       (macd < 0) & (macdsignal < 0) & (df['ema'][start:length] > df['ema'].shift(1)[start:length])
    sell_signal_first = (crossunder_series(df['macdhist'][max(start1 - 1, 0):length],
                                           0 * df['macdhist'][max(start1 - 1, 0):length], stream)) & \
                        (macd > 0) & (macdsignal > 0) & (df['ema'][start:length] < df['ema'].shift(1)[start:length])
    buy_signal, sell_signal, sell_price, stop_loss, sell_price_s, stop_loss_s, signal_tp = initialize_signals(
        buy_signal_first, sell_signal_first, df, start, length)

    for i in range(start, length):
        if buy_signal_first[i - start]:
            process_buy_signal(df, data, i, buy_signal, sell_price, stop_loss, signal_tp, start)
        if sell_signal_first[i - start]:
            process_sell_signal(df, data, i, sell_signal, sell_price_s, stop_loss_s, signal_tp, start)

    df.iloc[start:length, df.columns.get_loc('buy')] = buy_signal.astype(int)
    df.iloc[start:length, df.columns.get_loc('short')] = sell_signal.astype(int)
    df.iloc[start:length, df.columns.get_loc('turningpoints')] = signal_tp
    df.iloc[start:length, df.columns.get_loc('sell_price')] = sell_price
    df.iloc[start:length, df.columns.get_loc('stop_loss')] = stop_loss
    df.iloc[start:length, df.columns.get_loc('sell_price_s')] = sell_price_s
    df.iloc[start:length, df.columns.get_loc('stop_loss_s')] = stop_loss_s


def initialize_signals(buy_signal_first, sell_signal_first, df, start, length):
    """Initialize trading signals."""
    buy_signal = buy_signal_first * 0
    sell_signal = sell_signal_first * 0
    sell_price = (buy_signal_first * 0).astype(float)
    stop_loss = (buy_signal_first * 0).astype(float)
    sell_price_s = (buy_signal_first * 0).astype(float)
    stop_loss_s = (buy_signal_first * 0).astype(float)
    signal_tp = buy_signal * 0
    return buy_signal, sell_signal, sell_price, stop_loss, sell_price_s, stop_loss_s, signal_tp


def process_buy_signal(df, data, i, buy_signal, sell_price, stop_loss, signal_tp, start):
    """Process buy signals."""
    for j in range(i - 1, 0, -1):
        if df.sma_low[j - 1] > df.sma_low[j]:
            if df['in_sr_bull'][j] > 0:
                signal_tp[i - start] = int(i - j)
                sell_price[i - start] = float(df['in_sr_bull'][j])
                stop_loss[i - start] = float(df['close'][i] - (sell_price[i - start] - df['close'][i]))
                if stop_loss[i - start] > df['sl_bull'][j]:
                    stop_loss[i - start] = float(df['sl_bull'][j])
                if sell_price[i - start] < df['close'][i] or stop_loss[i - start] > df['close'][i]:
                    break
                if (sell_price[i - start] - df['close'][i]) / df['close'][i] < 0.001:
                    break
                buy_signal[i - start] = 1
                data.support_plot[i] = data.support[j]
                data.centers_plot[i] = data.sr_long[j]
                data.resistance_plot[i] = data.resistance[j]
            break
def process_sell_signal(df, data, i, sell_signal, sell_price_s, stop_loss_s, signal_tp, start):
    """Process sell signals."""
    for j in range(i - 1, 0, -1):
        if df.sma_high[j - 1] < df.sma_high[j]:
            if df['in_sr_bear'][j] > 0:
                in_sr_j = j
                signal_tp[i - start] = int(i - j)
                sell_price_s[i - start] = float(df['in_sr_bear'][j])
                stop_loss_s[i - start] = float(df['close'][i] + (df['close'][i] - sell_price_s[i - start]))
                if stop_loss_s[i - start] < df['sl_bear'][j]:
                    stop_loss_s[i - start] = float(df['sl_bear'][j])
                if sell_price_s[i - start] > df['close'][i] or stop_loss_s[i - start] < df['close'][i]:
                    break
                if (-sell_price_s[i - start] + df['close'][i]) / df['close'][i] < 0.001:
                    break
                sell_signal[i - start] = 1
                data.resistance_plot[i] = data.resistance[in_sr_j]
                data.centers_plot[i] = data.sr_long[in_sr_j]
                data.support_plot[i] = data.support[in_sr_j]
            break