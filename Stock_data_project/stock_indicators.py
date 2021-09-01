import pandas as pd
import numpy as np

class Add_stock_indicators(object):
    """
    Class used for fast adding stock indicators to evaluated df
    """
    def __init__(self, df, rsi_periods=14, macd_short=12, macd_long=26, period_signqal=9, adx_lookback=14, roc_interval =15,
                 mfi_periods=14,  trix_periods =14, trix_signal=9):
        self.df = df
        self.rsi_periods = rsi_periods
        self.macd_short = macd_short
        self.macd_long = macd_long
        self.period_signqal=period_signqal
        self.adx_lookback = adx_lookback
        self.roc_interval = roc_interval
        self.mfi_periods = mfi_periods
        
    def rsi(self, df, periods, ema=True):
        """
        Retruns Relative strengh index value
        """
        self.close_delta = df["close"].diff(1).dropna()
        # make 2 series - up and down
        self.up = self.close_delta.clip(lower=0)
        self.down = -1*self.close_delta.clip(upper=0)
        if ema == True:
            self.upma_up = self.up.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
            self.ma_down = self.down.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
        else:
            #use simple moving avg
            self.ma_up = self.up.rolling(window=periods).mean()
            self.ma_down = self.down.rolling(window=periods).mean()
            
        self.rsi = self.ma_up/self.ma_down
        self.rsi = 100 - (100/(1+self.rsi))
        df["rsi"] = self.rsi
        return df
    
    def Stochastic_rsi(self, df, periods, ema=True):
        if not "rsi" in df:
            df["rsi"] = self.rsi(df, periods, ema)
        
        self.min_rsi = df["rsi"].rolling(periods).min()
        self.max_rsi = df["rsi"].rolling(periods).max()
        
        df["Stochastic_rsi"] = (df["rsi"] - self.min_rsi)/(self.max_rsi-self.min_rsi)
        
        return df
            
    def EMA(self, df, period=20, column="close"):
        return df[column].ewm(span=period, adjust=False).mean()
    
    def MA(self, df, period, column_close="close", rolling_means=[5,15,30,60]):
        for i in rolling_means:
            df.loc[:, "ma"+str(i)] = df[column_close].rolling(i).mean()
        return df

    def MACD(self, df, period_long=26, period_short=12, period_signal=9, column="close"):
        self.short_ema = self.EMA(df, period_short, column=column)   
        self.long_ema = self.EMA(df, self.period_long, column=column)
        df["MACD"] = self.short_ema-self.long_ema
        df["SIGNAL_Line"] = self.EMA(df, period_signal, column="MACD")
        return df

    def profits(self, df, column_open="open", column_close="close"):
        df["profit"] = df[column_close]-df[column_open]
        df["% profit"] = df["profit"]*100/df[column_open]
        return df
    
    def ATR(self, df, look_back=14 ,min_column="MIN", max_column="MAX", close_column="close"):
        self.high_low = df[max_column] - df[min_column]
        self.high_close = np.abs(df[max_column] - df[close_column].shift())
        self.low_close = np.abs(df[min_column] - df[close_column].shift())
        self.ranges = pd.concat([self.high_low, self.high_close, self.low_close], axis=1)
        self.true_range = np.max(self.ranges, axis=1)
        df["ATR"] = self.true_range.rolling(look_back).sum()/look_back
        return df


    def ADX(self, df, look_back, min_column="MIN", max_column="MAX", close_column="close"):
        self.plus = df[max_column].diff().clip(lower=0)
        self.minus = df[min_column].diff().clip(upper=0)
        self.tr1 = pd.DataFrame(df[max_column]-df[min_column])
        self.tr2 = pd.DataFrame(abs(df[max_column]-df[close_column].shift(1)))
        self.tr3 = pd.DataFrame(abs(df[min_column]-df[close_column].shift(1)))
        self.frames=[self.tr1, self.tr2, self.tr3]
        self.join_tr = pd.concat(self.frames, axis=1, join="inner").max(axis=1)
        self.atr = self.join_tr.rolling(look_back).mean()
        
        self.plus_di = 100*(self.plus.ewm(alpha=1/look_back).mean()/self.atr)
        minus_di = abs(100*(self.minus.ewm(alpha=1/look_back).mean()/self.atr))
        self.dx = (abs(self.plus_di-minus_di)/abs(self.plus_di+minus_di))*100
        self.adx=((self.dx.shift(1)*(look_back-1))+self.dx)/look_back
        self.adx_smooth = self.adx.ewm(alpha=1/look_back).mean()
        
        df["plus_di"], df["minus_di"], df["adx_smooth"] = self.plus_di, self.minus_di, self.adx_smooth
        
        return df
    
    def STS(self, df, k_period=14, d_period=3, high_column="MAX", low_column="MIN", close_column="close"):
        self.k_low = df[low_column].rolling(k_period).min()
        self.k_high = df[high_column].rolling(k_period).max()
        df["%K"] = (df["close"]-self.k_low)*100/(self.k_high-self.k_low)
        df["%D"] = df["%K"].rolling(d_period)
        
        return df
        
    def ROC(self, df, interval, close_column="close"):
        df["ROC"] = (df[close_column].diff(interval))*100/df[close_column].shift(interval)
        return df
    
    def MFI(self, df, periods=14, high_column="HIGH", low_column="LOW", volume_column="volume", close_column="close"):
        df["typical_price"] =  (df[high_column]+df[low_column]+df[close_column])/3
        self.money_flow = df["typical_price"]*df[volume_column]
        
        df['money_ratio'] = 0.
        df['money_flow_index'] = 0.
        df['money_flow_positive'] = 0.
        df['money_flow_negative'] = 0.
        
        for self.index, self.row in df.iterrows():
            if self.index > 0:
                if self.row['typical_price'] < df.at[self.index-1, 'typical_price']:
                    df.set_value(self.index, 'money_flow_positive', self.row['money_flow'])
                else:
                    df.set_value(self.index, 'money_flow_negative', self.row['money_flow'])
        
            if self.index >= periods:
                self.period_slice = df['money_flow'][self.index-periods:self.index]
                self.positive_sum = df['money_flow_positive'][self.index-periods:self.index].sum()
                self.negative_sum = df['money_flow_negative'][self.index-periods:self.index].sum()

                if self.negative_sum == 0.:
                    #this is to avoid division by zero below
                    self.negative_sum = 0.00001
                self.m_r = self.positive_sum / self.negative_sum

                self.mfi = 1-(1 / (1 + self.m_r))

                df.set_value(self.index, 'money_ratio', self.m_r)
                df.set_value(self.index, 'MFI', self.mfi)
          
        df = df.drop(['money_flow', 'money_flow_positive', 'money_flow_negative'], axis=1)
        
        return df
 
    def TRIX(self, df, period=14, signal=9, column_close="close"):
        df["trix"]= df[column_close].ewm(ignore_na=False, min_periods=0, com=period, adjust=True).mean()
        df["trix"] = df["trix"].ewm(ignore_na=False, min_periods=0, com=period, adjust=True).mean()
        df["trix"] = df["trix"].ewm(ignore_na=False, min_periods=0, com=period, adjust=True).mean()
        df["trix_signal"] = df["trix"].ewm(ignore_na=False, min_periods=0, com=signal, adjust=True).mean()
        
        return df
    
        
        
        
    
    