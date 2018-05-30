import numpy as np
import pandas as pd
import scipy as sp
import os

from importlib import reload

from equation import *
from error_model import *

from control import *

from experiment import *

from time_series import *

from sampler import *

from het_model import *

from scipy.stats import norm

import os

from pathlib import Path

#parallel for multiple replicates
from mpi4py import MPI


comm = MPI.COMM_WORLD

numtask = MPI.COMM_WORLD.Get_size()
taskid = 	MPI.COMM_WORLD.Get_rank()


if taskid >= 0:
	#Read inference.dat and sim.dat:
	Input=control()

	TSE=TimeSeriesExperiment('A')
	print('read_string'+str(Input.Simulate_data))

	if(Input.Simulate_data): #Simulate data
		TSE.simulate_experiment(Input)
		TSE.write_file_experiment()
	else: #Read data file
		TSE.read_file_experiment(Input)

	Sampler = Input.sampler

	#Read existing MCMC sample file (out.dat) is present:
	Sampler.read_samples()




#Create a dictionnary to split replicates between tasks:
taskrep={}
keys = list(range(numtask))
keys.pop(0);
values = list(range(TSE.Nrep // (numtask - 1)) )
for itask in keys:
	taskrep[itask] = [x + (TSE.Nrep // (numtask - 1))*(itask-1) for x in values]

if(TSE.Nrep % (numtask - 1) > 0):
	taskrep[numtask-1].append(TSE.Nrep -1 )

if(taskid == 0):
	print('Replicates distributed according to the following dictionnary:',taskrep)

if 1:
	for isamp in range(Sampler.iter,Input.Nsamp):

		new_sample  = Sampler.param_new

		if(taskid == 0):

			while True:
				Sampler.step()
				new_sample = Sampler.param_new
				if(TSE.hetmodel.rate_model.check_bounds(TSE.hetmodel.get_rate(new_sample))):
					break
			
			loglik = 0.0


		new_sample = comm.bcast(new_sample, root=0)

		if(taskid > 0):
			loglik = 0
			for irep in (taskrep[taskid]):
				T1 = TSE.TS[irep]
				new_rates = TSE.hetmodel.rate_model.vary_coeff(TSE.hetmodel.get_rate(new_sample))
				T1 = TSE.TS[irep]
				T1.Equation.update_coeff(new_rates)

				new_error = TSE.hetmodel.error_model.vary_coeff(TSE.hetmodel.get_error_rep(new_sample,irep))
				T1.Error_model.update_coeff([new_error])

				T1 = TSE.TS[irep]
				ll = T1.calculate_likelihood_with_particle_filter(1000)
				loglik += ll
		
		#loglik= 1;
		sendmsg = loglik
		loglik = comm.reduce(sendmsg, op=MPI.SUM, root=0)


		if(taskid == 0):
			accept = Sampler.decide(loglik)
			if accept == 1:
				Sampler.save(loglik,Input)


exit
