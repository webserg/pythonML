import pandas as pd
import tensorflow as tf
import numpy as np
import pandas_datareader as pdr
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import random

random.seed(111)
rng = pd.date_range(start='2000' , end='2020')
ts = pd.Series(np.random.uniform(-10,10,size=len(rng)),rng).cumsum()
ts.plot(c='b', title ='test')
plt.show()
ts.head(10)