import numpy as np
from equation import *
from hetmodel import *
import os
import pathlib
import pandas as pd
import copy

class time_series(Equation):

	def __init__(self,X0,Ntime,Equation,Error_model,hetmodel,irep):
		
		self.Equation = Equation
		self.Error_model = Error_model
		self.hetmodel = hetmodel

		self.Ntime = Ntime;
		self.t = np.zeros(Ntime)
		self.x = np.zeros(Ntime)
		self.xh = np.zeros(Ntime)
		self.L = -1e99;
		self.X0 = X0
		self.irep = irep #index for the time series

	def initialize(self,t,x):
		self.x = x
		self.t = t
		self.X0 = x[0]


	def init_data(self,data):

		self.t = data.t
		self.x = data.x
		self.L = -1e99;
		self.X0 = data.x[0]

	def set_time(self,time):
		self.t = time

	def simulate_time_series(self):#Simulate data for time series:
		self.xh[0] = self.X0

		#Need to define a hidden and observed chains
		for i in range(1,self.Ntime):
			dt = self.t[i] - self.t[i-1]
			self.xh[i] = self.Equation.simulate_exact(self.xh[i-1],dt)

		#print('before',self.xh[1:10])
		self.x = np.round(self.Error_model.add_error(self.xh))
		#print('after',self.x[1:10])

	@classmethod

	def simulate_data(cls,irep,eq1,err1,hetmodel,X00,T,Ntime):
		#Simulate data:
			
			#Copy the equation here to not change the rates:
			eq = copy.deepcopy(eq1); err = copy.deepcopy(err1); 
			#print('coeff=',eq.coeff)
			eq.update_coeff(hetmodel.rate_model.vary_coeff(eq.coeff))
			err.update_coeff(hetmodel.error_model.vary_coeff(err.coeff))
			#print('after: coeff=',eq.coeff)
			T1 = time_series(X00,Ntime,eq,err1,hetmodel,irep)
			T1.set_time(np.linspace(0.0, T, num=T1.Ntime))
			T1.simulate_time_series()
			#d = {'t': T1.t, 'x': T1.x}
			#df = pd.DataFrame(data=d)
			#df.to_csv( 'data-1.dat',header=0,sep=' ', index=False)
			return T1



	def calculate_likelihood_with_particle_filter(self,Npart):#Calculate likelihood with particle filter:

		#Initialize particles at time -1 with a Poisson distribution and the same weights:
		y = np.random.poisson(self.x[0],Npart); w = np.repeat(1.0/Npart,Npart)

		sum_all = 0;
		for i in range(0,self.Ntime-2):#
			
			#Reweight particles:
			ynew = np.random.choice(y, size=Npart, replace=True, p=w)

			#Move particles:
			dt =  self.t[i+1] - self.t[i]

			#Use Euler method to propagate particles
			y = self.Equation.simulate_approx(ynew,dt)


			#Reweight particles:
			w = self.Error_model.error_likelihood(self.x[i+1],y)
			
			#Sum weights
			sum_w = np.sum(w)

			#Sum log (divide by number of particles to normalize the likelihood with the number of particles)
			sum_all += np.log(sum_w/Npart)

			w  = w/sum_w

		return sum_all


