import sys
import time
import datetime as dt
import multiprocessing
import ccxt
from live_trading_helpers import (
    set_leverage,
    gen_signals,
    run_search,
    adjust_leverage,
    set_limit_order,
    handle_trailing_stop,
    handle_new_signals,
    handle_open_orders,
    adjust_sleep_time,
    plot_trade
)
from config import SYMBOLS, BALANCE_ALLOCATION, DATA_WINDOW, TIMEFRAME, TIMEFRAME_SEC, SL_RATIO, TP_RATIO, LEVERAGE

# Configure the exchange
exchange = ccxt.bybit()
# exchange.set_sandbox_mode(True)  # enable sandbox mode

# Insert API KEY HERE
exchange.apiKey = 'XXXXXXXXXXXX'
exchange.secret = 'XXXXXXXXXXXXXXXXXXXXX'

if __name__ == "__main__":
    set_leverage(LEVERAGE, exchange, SYMBOLS)

    entry_price = {}
    position_strat = {}
    position_trail = {}
    position_HL = {}

    since = exchange.milliseconds() - DATA_WINDOW * TIMEFRAME_SEC * 1000
    data_dict = gen_signals(exchange, SYMBOLS, DATA_WINDOW, TIMEFRAME, since)
    queue1 = multiprocessing.Queue()
    proc1 = multiprocessing.Process(target=run_search, args=(queue1, data_dict, exchange))
    proc1.start()
    old_min = dt.datetime.now().minute
    last_bar = dt.datetime.now()

    while True:
        new_data = 0
        if not queue1.empty():
            data_dict = queue1.get()
            print('received')
            new_data = 1
            last_bar = dt.datetime.now()
            starting_balance = exchange.fetch_balance()['USDT']['total']

        for symbol in SYMBOLS:
            data = data_dict[symbol]
            in_position = 0
            positions = exchange.fetch_positions(symbols=[symbol])
            balance = exchange.fetch_balance()['USDT']['free']
            for position in positions:
                size = position['contracts']
                side = position['info']['side']
                lev = int(position['info']['leverage'])
                if size != 0:
                    in_position = 1
                    if position['info']['stop_loss'] == '0':  # Clear any positions without TP/SL
                        if side == 'Buy':
                            exchange.create_market_sell_order(symbol, size, params={'position_idx': 0})
                        else:
                            exchange.create_market_buy_order(symbol, size, params={'position_idx': 0})

                    # Re-adjust leverage based on free equity
                    leverage = adjust_leverage(balance, starting_balance, lev)
                    if lev > leverage:
                        exchange.set_leverage(leverage, symbol=symbol, params={})
                        print('reducing leverage on ', symbol)
                        balance = exchange.fetch_balance()['USDT']['free']

                    if len(exchange.fetch_open_orders(symbol)) == 0:  # Put limit order on newly opened position
                        set_limit_order(exchange, position, symbol, size, side, position_strat)
                        position_HL[symbol] = data.df['close'][-1]

                    if new_data and position_strat[symbol] == 1:
                        handle_trailing_stop(exchange, data, position, symbol, size, side, entry_price, position_HL, position_trail)

            if new_data:
                handle_new_signals(exchange, data, symbol, entry_price, position_strat, position_trail, in_position, balance, starting_balance)

        handle_open_orders(exchange, new_data, last_bar, entry_price)

        now = dt.datetime.now()
        sys.stdout.flush()
        adjust_sleep_time(now, last_bar)