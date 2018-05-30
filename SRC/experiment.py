from control import *
from time_series import *
from equation import *
from error_model import *
import numpy as np



class Experiment():


	def __init__(self,Nrep,Type,Name):
		self.Nrep= Nrep #Number of replicates
		self.Type = Type #Type of experiment
		self.Name = Name

class TimeSeriesExperiment(Experiment):

	def __init__(self,name1):
		Experiment.__init__(self,1,'Time Series',name1)
	

	def simulate_experiment(self,control):
		
		self.Nrep = control.Nrep_sim
		print('sdssd',self.Nrep)
		#Conditions for the start conditions:
		if self.Nrep == control.X0_sim.size:
			IC = control.X0_sim
		elif control.X0_sim.size == 1:
			IC = np.random.poisson(control.X0_sim[0],control.Nrep_sim)
		else:
			print("Wrong size for initial conditions for simulated data")

		print(IC[0])

		#Create list of time series with all the replicates:
		TS = []
		for irep in range(self.Nrep ):
			print('simulating'+str(irep))
			T1 = time_series.simulate_data(irep,control.Equation_sim,control.Error_sim,control.hetmodel_sim,IC[irep],control.T_sim,control.Ntime_sim)
			print(T1.Ntime)
			TS.append(T1)

		self.TS = TS
		self.hetmodel = control.hetmodel#inherit the het model of the simulation

	def write_file_experiment(self):

		print('Writing simulated data to file')

		appended_data = []

		for irep in range(self.Nrep):

			TS = self.TS[irep]
			df=pd.DataFrame({'t':TS.t, 'x':TS.x, 'replicate':np.repeat(irep,TS.t.size)})

			appended_data.append(df)

		appended_data = pd.concat(appended_data, axis=0)

		appended_data.to_csv(data_file_name)

		print(appended_data)
		#Create a data frame now:
		#Time value, replicate, type, Experiment name:



	def read_file_experiment(self,Input):

		print('Reading data file:'+data_file_name)
		df_read = pd.read_csv(data_file_name, index_col=0)

		#Count the number of replicates:
		allrep = df_read.replicate.unique()
		self.Nrep = df_read.replicate.unique().size

		#Check that the error model is properly specified:
		#if(Input.param_error.size == 1 & Input.param_error.size)

		#Create list of time series with all the replicates:
		TS = []
		for irep in allrep:
			print('reading replicate '+str(irep))
			df_rep = df_read[df_read.replicate == irep]
			t = np.array(df_rep.t.values); x = np.array(df_rep.x.values); 
			T1 = time_series(x[0],x.size,Input.equation,Input.error_model,Input.hetmodel,irep)
			T1.initialize(t,x)
			TS.append(T1)

		self.TS = TS
		self.hetmodel = Input.hetmodel



