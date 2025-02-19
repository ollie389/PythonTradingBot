# config.py

import Crypto_Strategy_Kmean

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOTUSDT', 'LTCUSDT', 'LINKUSDT', 'ETCUSDT']
BALANCE_ALLOCATION = 0.5
DATA_WINDOW = 800
TIMEFRAME = '5m'
TIMEFRAME_SEC = 5 * 60  # Must be same as above timeframe but in seconds
SL_RATIO = 1.2 / 1.25
TP_RATIO = 1.25
LEVERAGE = 1

# Initialize strategy
STRATEGY = Crypto_Strategy_Kmean.strategy