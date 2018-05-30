from het_model import *

class hetmodel:

	def __init__(self):
		self.Ndim = 0 #This is the number of parameters describing the rates
		#In the case of homogeneous rate model, equal to the number of parameters

	def set_rate_model(self,rate_model,Ndim):
		self.rate_model = rate_model
		self.Ndim = Ndim

	def set_error_model(self,error_model):
		self.error_model = error_model

	def get_rate(self,coeff):
		return(coeff[0:self.Ndim])

	def get_error(self,coeff):
		return(coeff[self.Ndim:coeff.size])

	def get_error_rep(self,coeff,irep):
		return(coeff[self.Ndim:coeff.size][irep])	