import matplotlib.pyplot as plt
from scipy.stats import norm
data_normal = norm.rvs(size=10000,loc=4,scale=1)
# import seaborn
import seaborn as sns
# settings for seaborn plotting style
sns.set(color_codes=True)
# settings for seaborn plot sizes
sns.set(rc={'figure.figsize':(5,5)})
sns.distplot(data_normal,
             bins=100,
             kde=True,
             color='skyblue',
             hist_kws={"linewidth": 15,'alpha':1})

plt.show()