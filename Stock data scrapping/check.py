import sqlite3
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.graph_objs as go 
import torch


def connect_db(path):
    db = sqlite3.connect(path)
    return db

def rsi(df, periods=14, ema=True):
    """
    Retruns Relative strengh index value
    """
    close_delta = df["close"].diff(1).dropna()
    # make 2 series - up and down
    up = close_delta.clip(lower=0)
    down = -1*close_delta.clip(upper=0)
    if ema == True:
        ma_up = up.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
        ma_down = down.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
    else:
        #use simple moving avg
        ma_up = up.rolling(window=periods).mean()
        ma_down = down.rolling(window=periods).mean()
        
    rsi = ma_up/ma_down
    rsi = 100 - (100/(1+rsi))
    
    return rsi

def EMA(df, period=20, column="close"):
    return df[column].ewm(span=period, adjust=False).mean()

def MACD(df, period_long=26, period_short=12, period_signal=9, column="close"):
    short_ema = EMA(df, period_short, column=column)   
    long_ema = EMA(df, period_long, column=column)
    df["MACD"] = short_ema-long_ema
    df["SIGNAL_Line"] = EMA(df, period_signal, column="MACD")
    return df

def profits(df, column_open="open", column_close="close"):
    df["profit"] = df[column_close]-df[column_open]
    df["% profit"] = df["profit"]*100/df[column_open]
    return df
 
def ATR(df, look_back=14 ,min_column="MIN", max_column="MAX", close_column="close"):
    high_low = df[max_column] - df[min_column]
    high_close = np.abs(df[max_column] - df[close_column].shift())
    low_close = np.abs(df[min_column] - df[close_column].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df["ATR"] = true_range.rolling(look_back).sum()/look_back
    return df


def get_adx(df, look_back, min_column="MIN", max_column="MAX", close_column="close"):
    plus = df[max_column].diff().clip(lower=0)
    minus = df[min_column].diff().clip(upper=0)
    tr1 = pd.DataFrame(df[max_column]-df[min_column])
    tr2 = pd.DataFrame(abs(df[max_column]-df[close_column].shift(1)))
    tr3 = pd.DataFrame(abs(df[min_column]-df[close_column].shift(1)))
    frames=[tr1, tr2, tr3]
    join_tr = pd.concat(frames, axis=1, join="inner").max(axis=1)
    atr = join_tr.rolling(look_back).mean()
    
    plus_di = 100*(plus.ewm(alpha=1/look_back).mean()/atr)
    minus_di = abs(100*(minus.ewm(alpha=1/look_back).mean()/atr))
    dx = (abs(plus_di-minus_di)/abs(plus_di+minus_di))*100
    adx=((dx.shift(1)*(look_back-1))+dx)/look_back
    adx_smooth = adx.ewm(alpha=1/look_back).mean()
    df["plus_di"], df["minus_di"], df["adx_smooth"] = plus_di, minus_di, adx_smooth
    
    return df

def partition_dataset(sequence_length, df):
    x, y = [], []
    data_len = df.shape[0]
    index_Close = df.columns.get_loc("close")
    for i in range(sequence_length, data_len):
        x.append(df[i-sequence_length:i,:]) #contains sequence_length values 0-sequence_length * columsn
        y.append(df[i, index_Close]) #contains the prediction values for validation,  for single-step prediction
    
    # Convert the x and y to numpy arrays
    x = np.array(x)
    y = np.array(y)
    return x, y

if __name__ == '__main__':
    cwd = os.getcwd()
    data_base_path = os.path.join(cwd, "orlen_stock_v2.db")
    con = connect_db(data_base_path)
    df = pd.read_sql_query("SELECT * from pkn_stock_v2 ORDER by date ASC", con)
    df = df.set_index("date")

    rolling_means = [5,10,20,50,100]
    for i in rolling_means:
        df.loc[:, "ma"+str(i)] = df.close.rolling(i).mean()

    df = MACD(df)
    df = ATR(df)
    df = get_adx(df, look_back=14)
    df = profits(df)

    df["rsi"] = rsi(df)

    dataset = df.dropna().drop(columns=["open", "MIN", "MAX"])
    print(dataset.head())

    train_split = int(dataset.shape[0]*0.90)
    train_data = dataset[:train_split]
    test_data = dataset[train_split:]
    