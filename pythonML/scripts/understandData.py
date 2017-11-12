# Обеспечим совместимость с Python 2 и 3
# pip install future
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

# отключим предупреждения Anaconda
import warnings
warnings.simplefilter('ignore')

# импортируем Pandas и Numpy
import pandas as pd
import numpy as np

df = pd.read_csv('./data/telecom_churn.csv')
print(df.head())