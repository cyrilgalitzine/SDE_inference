import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df_in = pd.read_csv('out.dat',sep=' ')

frequency = 400
istart = 10000

df = df_in.iloc[istart::frequency,:]


#Get list of variables
variables = list(df); variables.remove('iter'); variables.remove('L')

for var1 in variables:
	#Plot evolution of variables:
	plt.figure()
	plt.plot(df_in.iter,df_in.eval(var1))
	plt.plot(df.iter,df.eval(var1),'o')
	plt.savefig(var1+'_evol.pdf')
	plt.close()	


#Calculate autocorrelation spectrum:
for var1 in variables:
	y = (df.eval(var1) - np.mean(df.eval(var1)))/np.std(df.eval(var1))
	correlated = np.correlate(y, y,  mode='full')
	spec=correlated[int((correlated.size - 1)/2):correlated.size]/len(y)
	xs = range(0,spec.size)
	plt.figure()
	plt.bar(xs[0:30],spec[0:30],width=0.4)
	plt.savefig(var1+'_corr.pdf')
	plt.close()

#Plot density:
for var in variables:
	plt.figure()
	lab= var+' density, Niter='+ str(df.shape[1])
	if(var == 'kd'):
		sns.kdeplot(df.eval(var), label=var, bw = 5e-4)
	else:	
		sns.kdeplot(df.eval(var), label=var)
	plt.plot()
	plt.savefig(var+'_density.pdf')
	plt.close()


#Output density values to file:
