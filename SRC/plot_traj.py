import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data.csv')
df1 = df[['t','x','replicate']]
plt.figure()
sns.lmplot(data = df1,x='t',y='x',hue='replicate',fit_reg =False)


plt.savefig('data.png')

