import numpy as np
from scipy.stats import norm
#Define an equation class

class Error_model:

	def __init__(self,name,coeff):
		self.name = name
		self.coeff = coeff

	def update_coeff(self,coeff):
		self.coeff = coeff

#Define particular equation class from general equation class
class Normal_error_model(Error_model):

	def __init__(self,coeff):
		Error_model.__init__(self,'normal',coeff)

	def error_likelihood(self,x,y):

		coeff = self.coeff
		sd = coeff[0]
		f = norm.pdf(y,loc=x, scale=sd)
		return f

	def add_error(self,x):

		coeff = self.coeff
		sd = coeff[0]
		f = np.random.normal(loc=x, scale=sd)
		return f

	#def update_error(self,coeff):
	#	super(Normal_error_model,self).update_error((self,coeff))


