from gym.envs.registration import register
from .portfolio import PortfolioEnv
import pandas as pd

df_train = pd.read_csv('/home/mateusz/Desktop/Moje_repo/My-Data-Science-repository/Stock_data_project/mWIG40/WIG40/portfolio optimisation/train_WIG_data.csv')

env_specs_args = [
    dict(id='TradingEnv-v1',
         entry_point='portfolio:PortfolioEnv',
         kwargs=dict(
             output_mode='mlp',
             df=df_train
         ))]


env_specs = [spec['id'] for spec in env_specs_args]
# register our env's on import
for env_spec_args in env_specs_args:
    register(**env_spec_args)