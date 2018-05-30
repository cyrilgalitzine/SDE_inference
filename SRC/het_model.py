import numpy as np

class het_model:

	def __init__(self,Nrep,name):
		self.name = name
		self.Nrep = Nrep
		


class homogeneous_het_model(het_model):

	def __init__(self):
		het_model.__init__(self,1,'homogeneous')

	def get_coeff(self,coeff):
		return coeff

	def vary_coeff(self,coeff):
		return coeff

	def check_bounds(self,coeff): #bound check not necessary.
		return 1

class heterogeneous_het_model(het_model):

	def __init__(self):
		het_model.__init__(self,1,'heterogeneous')

	def get_coeff(self,coeff):
		return coeff

	def check_bounds(self,coeff):#Limit the ratio of sd to mean to less than tol = 5
	#This avoid issues with the gamma distribution number generator
		mean1 = coeff[0::2]
		sd1 =	 coeff[1::2]
		out = 1
		for i in range(mean1.size):
			if(sd1[i]/mean1[i] > 5):
				out = 0
		return out



	def vary_coeff(self,coeff): #Need to change this to account for gamma distribution
		mean1 = coeff[0::2]
		sd1 =	 coeff[1::2]
		var1 = np.power(sd1,2.0)

		scale = var1/mean1
		shape = np.power(mean1,2)/var1

		s = np.random.gamma(shape, scale)
		return s

