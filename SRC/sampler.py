import numpy as np
from scipy.stats import norm
import pandas as pd
import pathlib
#Define an equation class

class sampler:

	def __init__(self,name,param,param_infer,sd,output_freq):
		self.name = name
		self.param_new = param #has to be a numpy array
		self.param_old = param
		self.Ndim = param.size #Should always be set equal to the total number of parameters (inferred + non-inferred)
		self.param_accept_stat =np.zeros(self.Ndim)
		self.param_infer = param_infer
		self.sd = sd #Standard deviation

		self.iter=-1
		self.max = -1e99

		self.alpha = 0.001

		self.freq = output_freq
		self.param_store = np.zeros(shape=(self.freq,self.Ndim))
		self.Lstore = np.zeros(shape=(self.freq,1))
		self.iterstore = np.zeros(self.freq)

	def save(self,L,Input):
		#print((np.matrix(self.iterstore).T.astype(int)))
		#print(np.concatenate((np.matrix(self.iterstore).T.astype(int),self.param_store,self.Lstore),axis=1))
		index = self.iter%self.freq
		self.param_store[index,:]= self.param_new
		self.Lstore[index,0]= L
		self.iterstore[index] = self.iter

		if(index == self.freq -1):

			header1 = " ".join(['iter'," ".join(Input.param_name)," ".join(Input.error_name),'L'])
			if(self.iterstore[0]>0):
				header1 =''

			f=open('out.dat','ab')
			np.savetxt(f,np.concatenate((np.matrix(self.iterstore).T.astype(int),self.param_store,self.Lstore),axis=1),header= header1,comments='')
			f.close()

	def read_samples(self):

		my_file = pathlib.Path('out.dat')

		if my_file.is_file():
			df_in = pd.read_csv('out.dat',sep=' ')

		else: 
			print('restart data not found')
			return

		last_line = df_in.iloc[-1]
		self.iter = np.int(last_line.iter)
		self.max = last_line.L
		self.param_old = np.array(last_line[1:last_line.size-1])



class MH_logexp(sampler):

	def __init__(self,param,param_infer,sd,output_freq):
		sampler.__init__(self,'MH_logexp',param,param_infer,sd,output_freq)
		print("initial with",param)


	def step(self):
		#Allows for some parameters to be constant:
		self.param_new = np.where(self.param_infer,self.param_old*np.exp(np.random.normal(loc = 0, scale = self.sd,size=self.Ndim)),self.param_old)

	def decide(self,L):
		accept = 0
		self.param_accept_stat *= (1 -self.alpha)
		frac = L - self.max - np.log(np.prod(self.param_old)) + np.log(np.prod(self.param_new))
		#print('frac=',np.log(np.prod(self.param_old)) - np.log(np.prod(self.param_new)))
		#print((self.param_new))
		P = min(frac,1.0)

		#print('L=',L,self.max)

		if( P > np.log(np.random.uniform()) ):
			print(self.param_new)
			print('L=',L,'Lmin=',self.max,'AR=',self.param_accept_stat)
			self.param_old = self.param_new
			self.max = L
			accept = 1
			self.param_accept_stat += self.alpha
			self.iter +=1

		return(accept)


