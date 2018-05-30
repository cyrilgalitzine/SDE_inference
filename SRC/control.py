import numpy as np
import pathlib
from equation import *
from error_model import *
from het_model import *
from hetmodel import *
from sampler import *
import os.path
data_file_name = "data.csv"

class control:

	def __init__(self):

		self.simulate = 0

		#Read inference.dat file which controls the inference procedure:
		with open('inference.dat', 'r') as f:
			for line in f:
				s = line
				x = s.strip('\n'); x = x.split(' ');

				if x[0] == "Equation":#Set equation model
					input_read = x[1]
					if(input_read == "BDI"):
						self.equation = BDI_Equation([[0.0,0.0,0.0]])

				if x[0] == "Ndisc":#Update number of discretization points for Euler scheme
					print('Ndisc=',int(x[1]))
					input_read = int(x[1])
					self.equation.update_Ndisc(input_read)

				elif x[0] == "Error_model":#Set error model
					input_read = x[1]
					if(input_read == "Normal"):	
						self.error_model = Normal_error_model([0.0])

				elif x[0] == "Nsamp":
					self.Nsamp = int(x[1])

				elif x[0] == "Npart":
					self.Npart = int(x[1])

				elif x[0] == "Simulate_data":#Simulate data before inference?
					input_read = int(x[1])
					if(input_read == 0): #Read data
						if(os.path.isfile(data_file_name) is True): #Test for presence of data file
							self.Simulate_data = 0
						else: #Simulate data since file not there
							print(data_file_name+'file not found, simulating data')
							self.Simulate_data = 1
					else:
						self.Simulate_data = 1
				elif x[0] == "param_infer": #Parameters that are to be inferred: (1=Yes,0=No)
					param = x[1]
					param = param.strip('[]'); param = param.split(',')

					param = [float(i) for i in param];
					
					self.param_infer = np.array(param)
					#print('####'+as.string(self.param_infer))
				elif x[0] == "param": #Starting value for the inferred parameters:
					param = x[1]
					param = param.strip('[]'); param = param.split(',')


					param_name = [i.split('=')[0] for i in param]
					param_val = [float(i.split('=')[1]) for i in param]

					self.starting_param = np.array(param_val)
					self.param_name = param_name
					self.equation.update_coeff(self.starting_param )

				elif x[0] == "sd_MH": #Set MH hastings standard deviation
					self.sd_MH = float(x[1])
				elif x[0] == "param_error": #Update the value of the error parameters:
					param = x[1]
					param = param.strip('[]'); param = param.split(',')

					param_name = [i.split('=')[0] for i in param]
					param_val = [float(i.split('=')[1]) for i in param]					
					
					self.starting_error = np.array(param_val)
					self.error_name = param_name
					self.error_model.update_coeff(self.starting_error)

				elif x[0] == "param_error_infer": #Infer yes or no the value of the error (1=Yes,0=No)
					param = x[1]

					param = param.strip('[]'); param = param.split(',')
					param = [int(i) for i in param]; self.param_error_infer = param#np.array(bool(param))
					print('@@@',np.array(bool(param)))

				elif x[0] == "Rate_heterogeneity_model":
					self.hetmodel = hetmodel()
					input_read = x[1]
					print(input_read)
					if(input_read == "Homogeneous"):
						hom_rate = homogeneous_het_model()
						self.hetmodel.set_rate_model(hom_rate,self.starting_param.size)
					if(input_read == "Heterogeneous"):
						hom_rate = heterogeneous_het_model()
						self.hetmodel.set_rate_model(hom_rate,self.starting_param.size)						

				elif x[0] == "Error_heterogeneity_model":
					input_read = x[1]
					print(input_read)
					if(input_read == "Homogeneous"):
						hom_error = homogeneous_het_model()
						self.hetmodel.set_error_model(hom_error)
					if(input_read == "Heterogeneous"):
						hom_error = heterogeneous_het_model()
						self.hetmodel.set_error_model(hom_error)
						#Need to change the error parameters now (one for each replicate):
						 	 

				elif x[0] == "MH_step_scaling":#Set model for scaling of parameters with Metropolis hastings:
					input_read = x[1]
					if(input_read == "exponential"):
						start_all = np.append(self.starting_param, self.starting_error)
						print("start_all",start_all)
						infer_all = np.append(self.param_infer, self.param_error_infer)
						print('infer_all',infer_all)
						self.sampler = MH_logexp(start_all,infer_all,self.sd_MH,10)										


		my_file = pathlib.Path("sim.dat")#To simulate data:
		if my_file.is_file():
			self.simulate = 1
			with open('sim.dat', 'r') as f:
				for line in f:
					s = line
					x = s.strip('\n'); x = x.split(' ');
					if x[0] == "param_sim":
						param = x[1]
						param = param.strip('[]'); param = param.split(',')
						param = [float(i) for i in param]; self.param_sim = np.array(param)
						self.Equation_sim.update_coeff(self.param_sim)

					elif x[0] == "error_sim":
						param = x[1]
						param = param.strip('[]'); param = param.split(',')
						param = [float(i) for i in param]; self.param_error = np.array(param)	
						self.Error_sim.update_coeff(self.param_error)

					elif x[0] == "Equation_sim":
						input_read = x[1]
						print(input_read)
						if(input_read == "BDI"):
							self.Equation_sim = BDI_Equation([0.0,0.0,0.0])

					elif x[0] == "Error_model_sim":
						input_read = x[1]
						if(input_read == "Normal"):	
							self.Error_sim = Normal_error_model([0.0])

					elif x[0] == "Nrep_sim":
						self.Nrep_sim = int(x[1])

					elif x[0] == "Ntime_sim":
						self.Ntime_sim = int(x[1])

					elif x[0] == "T_sim":
						self.T_sim = int(x[1])

					elif x[0] == "X0_sim":
						param = x[1]
						param = param.strip('[]'); param = param.split(',')
						param = [float(i) for i in param]; self.X0_sim = np.array(param)

					elif x[0] == "Rate_heterogeneity_model_sim":
						self.hetmodel_sim = hetmodel()
						input_read = x[1]
						print(input_read)
						if(input_read == "Homogeneous"):
							hom_rate_sim = homogeneous_het_model()
							self.hetmodel_sim.set_rate_model(hom_rate_sim,self.param_sim.size)

						if(input_read == "Heterogeneous"):
							hom_rate_sim = heterogeneous_het_model()
							self.hetmodel_sim.set_rate_model(hom_rate_sim,self.param_sim.size)

					elif x[0] == "Error_heterogeneity_model_sim":
						input_read = x[1]
						print(input_read)
						if(input_read == "Homogeneous"):
							hom_error_sim = homogeneous_het_model()
							self.hetmodel_sim.set_error_model(hom_error_sim) 


		else:
			print("sim.dat not found")



		#Need to properly initialize heterogeneous model:
  




		
