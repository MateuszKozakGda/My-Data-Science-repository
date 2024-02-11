import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime
import os
import logging
from pprint import pprint

import gym
import gym.spaces

logger = logging.getLogger(__name__)


class DataProvider(object):
    """
    Acts as data provider to Enviroment
    https://github.com/wassname/rl-portfolio-management/blob/master/rl_portfolio_management/environments/portfolio.py
    """
    def __init__(self, df, steps=50, scale=True, scale_extra_cols=True, 
                 augment=0.001, window_lengh=50, random_reset=True):
        """
            df - csv for data frame index of timestamps
                and multi-index columns levels=[['LTCBTC'],...],['open','low','high','close',...]]
                an example is included as an hdf file in this repository
            steps - total steps in episode
            scale - scale the data for each episode
            scale_extra_cols - scale extra columns by global mean and std
            augment - fraction to augment the data by
            random_reset - reset to a random time (otherwise continue through time)
        """
        self.steps = steps + 1
        self.augment = augment
        self.random_reset = random_reset
        self.scale = scale
        self.scale_extra_cols = scale_extra_cols
        self.window_length = window_lengh
        self.idx = self.window_length
        
        #dataframe to matrix transfomration
        self.asset_names = df.columns.levels[0].tolist()
        self.features = df.columns.levels[1].tolist()
        data = df.values.reshape(
            (len(df), len(self.asset_names), len(self.features)))
        self._data = np.transpose(data, (1, 0, 2)) ## transponse to: (Num Asset, len(df), Features)
        self._times = df.index

        self.price_columns = ["open", "MAX", "MIN", "close"]
        self.non_price_columns = set(
            df.columns.levels[1]) - set(self.price_columns)
        
        if scale_extra_cols:
            x = self._data.reshape((-1, len(self.features)))
            self.stats = dict(mean=x.mean(0), std=x.std(0)) ## mean and std used for MIN/MAX scalling
        
        self.reset()

    def _step(self):
        """
        https://arxiv.org/pdf/1706.10059.pdf 
        """
        # get history matrix from dataframe
        data_window = self.data[:, self.step:self.step + self.window_length].copy()

        # (eq.1) prices
        y1 = data_window[:, -1, 3] / data_window[:, -2, 3] ## eq(1) get close price (column index 3)
        y1 = np.concatenate([[1.0], y1])  # add cash price

        # (eq 18) X: prices are divided by close price
        nb_pc = len(self.price_columns) #number of price columns -> 4
        
        if self.scale:
            last_close_price = data_window[:, -1, 0]
            data_window[:, :, :nb_pc] /= last_close_price[:, np.newaxis, np.newaxis] #divide by last close price in batch
        
        if self.scale_extra_cols:
            # normalize non price columns -> column "trade"
            data_window[:, :, nb_pc:] -= self.stats["mean"][None, None, nb_pc:]
            data_window[:, :, nb_pc:] /= self.stats["std"][None, None, nb_pc:]
            data_window[:, :, nb_pc:] = np.clip(
                data_window[:, :, nb_pc:],
                self.stats["mean"][nb_pc:] - self.stats["std"][nb_pc:] * 10,
                self.stats["mean"][nb_pc:] + self.stats["std"][nb_pc:] * 10
            )

        self.step += 1
        history = data_window
        done = bool(self.step >= self.steps)
        
        return history, y1, done
    
    def reset(self):
        
        self.step = 0
        
        # get data for this episode
        if self.random_reset: #get random period
            self.idx = np.random.randint(
                low=self.window_length + 1, high=self._data.shape[1] - self.steps - 2)
                #DATA must be in range -> we have to have no less then window lengh data points
                #end point index should be lower then len(df)-steps
        else:
            # continue sequentially, before reseting to start
            if self.idx>(self._data.shape[1] - self.steps - self.window_length - 1):
                self.idx=self.window_length + 1
            else:
                self.idx += self.steps
                
        data = self._data[:, self.idx -
                          self.window_length:self.idx + self.steps + 1].copy()
        self.times = self._times[self.idx -
                                 self.window_length:self.idx + self.steps + 1]

        # augment data to prevent overfitting
        data += np.random.normal(loc=0, scale=self.augment, size=data.shape)

        self.data = data
        

class PortfolioSim(object):
    """
    Portfolio management sim.
    Params:
    - cost e.g. 5 zl -> bank comission
    Based of [Jiang 2017](https://arxiv.org/abs/1706.10059)
    - time cost - cost for maintaining the traiding account
    """

    def __init__(self, asset_names=[], steps=50, traiding_cost=5, time_cost=0.0):
        self.traiding_cost = traiding_cost
        self.time_cost = time_cost
        self.steps = steps
        self.asset_names = asset_names
        self.eps = 1e-8
        self.reset()
    
    def _step(self, w1, y1):
        """
        Step.
        w1 - new action of portfolio weights - e.g. [0.1,0.9, 0.0]
        y1 - price relative vector also called return
            e.g. [1.0, 0.9, 1.1]
        Numbered equations are from https://arxiv.org/abs/1706.10059
        """
        w0 = self.w0
        p0 = self.p0
        # eq7
        dw1 = (y1 * w0) / (np.dot(y1, w0) + self.eps) # adding eps for avoiding divide by 0

        # (eq16) cost to change portfolio
        # (excluding change in cash [1:] to avoid double counting for transaction cost)
        c1 = self.cost * (np.abs(dw1[1:] - w1[1:])).sum()
        p1 = p0 * (1 - c1) * np.dot(y1, w0)  # (eq11) final portfolio value
        p1 = p1 * (1 - self.time_cost) # add cost of holding assets
        #clipping -> cannot have negative holds of stocks
        p1 = np.clip(p1, 0, np.inf)
        rho1 = p1 / p0 - 1  # rate of returns
        r1 = np.log((p1 + self.eps) / (p0 + self.eps))  # (eq10) log rate of return
        # (eq22) immediate reward is log rate of return scaled by episode length
        reward = r1 / self.steps
        
        # remember for next step
        self.w0 = w1
        self.p0 = p1
        
        # if we run out of money, we're done
        done = bool(p1 == 0)
        
                # should only return single values, not list
        info = {
            "reward": reward,
            "log_return": r1,
            "portfolio_value": p1,
            "market_return": y1.mean(),
            "rate_of_return": rho1,
            "weights_mean": w1.mean(),
            "weights_std": w1.std(),
            "cost": c1,
        }
        
        
        for i, name in enumerate(["CASH"] + self.asset_names):
            info['weight_' + name] = w1[i]
            info['price_' + name] = y1[i]

        self.infos.append(info)
        return reward, info, done
    
    
    def reset(self):
        self.infos = []
        self.w0 = np.array([1.0] + [0.0] * len(self.asset_names)) # 1.0 is a cash plaeceholder
        self.p0 = 1.0
        
        
class PortfolioEnv(gym.Env):
    """
    An environment for financial portfolio management.
    Financial portfolio management is the process of constant redistribution of a fund into different
    financial products.
    Based on [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    metadata = {'render.modes': ['notebook', 'ansi']}
    
    def __init__(self,
                 df,
                 steps=50,
                 trading_cost=5,
                 time_cost=0.00,
                 window_length=50,
                 augment=0.001,
                 output_mode='mlp',
                 log_dir=None,
                 scale=True,
                 scale_extra_cols=True,
                 random_reset=True
                 ):
        

        """
        An environment for financial portfolio management.
        Params:
            df - csv for data frame index of timestamps
                 and multi-index columns levels=[[ASSET NAME],...],['open','low','high','close']]
            steps - steps in episode
            window_length - how many past observations["history"] to return
            trading_cost - cost of trade as a fraction -> 5zl
            time_cost - cost of holding as a fraction
            augment - fraction to randomly shift data by
            output_mode: decides observation["history"] shape
            - 'EIIE' for (assets, window, 3)
            - 'mlp' for (assets*window*3)
            log_dir: directory to save plots to
            scale - scales price data by last opening price on each episode (except return)
            scale_extra_cols - scales non price data using mean and std for whole dataset
        """
        self.window_lengh = window_length
        self.src = DataProvider(df=df, steps=steps, scale=scale, scale_extra_cols=scale_extra_cols,
                           augment=augment, window_lengh=self.window_lengh, random_reset=random_reset)
        
        self._plot = self._plot2 = self._plot3 = None
        self.output_mode = output_mode
        self.sim = PortfolioSim(
            asset_names=self.src.asset_names,
            trading_cost=trading_cost,
            time_cost=time_cost,
            steps=steps)
        self.log_dir = log_dir
        self.eps = 1e-8

        # openai gym attributes
        # action will be the portfolio weights [cash_bias,w1,w2...] where wn are [0, 1] for each asset
        nb_assets = len(self.src.asset_names)
        self.action_space = gym.spaces.Box(
            0.0, 1.0, shape=nb_assets + 1)

        # get the history space from the data min and max
        if output_mode == 'EIIE':
            obs_shape = (
                nb_assets,
                window_length,
                len(self.src.features)
            )
            
        elif output_mode == 'mlp':
            obs_shape = (nb_assets) * window_length * \
                (len(self.src.features))
        else:
            raise Exception('Invalid value for output_mode: %s' %
                            self.output_mode)

        self.observation_space = gym.spaces.Dict({
            'history': gym.spaces.Box(
                -10,
                20 if scale else 1,  # if scale=True observed price changes return could be large fractions
                obs_shape
            ),
            'weights': self.action_space
        })
        self._reset()
    
    def _step(self, action):
        """
        Step the env.
        Actions should be portfolio [w0...]
        - Where wn is a portfolio weight between 0 and 1. The first (w0) is cash_bias
        - cn is the portfolio conversion weights see PortfolioSim._step for description
        """
        logger.debug('action: %s', action)

        weights = np.clip(action, 0.0, 1.0)
        weights /= weights.sum() + self.eps

        # Sanity checks
        assert self.action_space.contains(
            action), 'action should be within %r but is %r' % (self.action_space, action)
        np.testing.assert_almost_equal(
            np.sum(weights), 1.0, 3, err_msg='weights should sum to 1. action="%s"' % weights)

        history, y1, done1 = self.src._step()

        reward, info, done2 = self.sim._step(weights, y1)

        # calculate return for buy and hold a bit of each asset
        info['market_value'] = np.cumprod(
            [inf["market_return"] for inf in self.infos + [info]])[-1]
        # add dates
        info['date'] = self.src.times[self.src.step].timestamp()
        info['steps'] = self.src.step

        self.infos.append(info)

        # reshape history according to output mode
        if self.output_mode == 'EIIE':
            pass
        elif self.output_mode == 'mlp':
            history = history.flatten()

        return {'history': history, 'weights': weights}, reward, done1 or done2, info
       
    def _reset(self):
        self.sim.reset()
        self.src.reset()
        self.infos = []
        action = self.sim.w0
        observation, reward, done, info = self.step(action)
        return observation
    
    def _render(self, mode='ansi', close=False):
        # if close:
            # return
        if mode == 'ansi':
            pprint(self.infos[-1])