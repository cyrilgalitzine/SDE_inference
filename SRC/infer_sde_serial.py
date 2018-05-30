import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import scipy as sp
#import seaborn as sns

#from importlib import reload

from equation import *
from error_model import *

from control import *

from experiment import *

from time_series import *

from sampler import *

from het_model import *

from scipy.stats import norm

import pathlib



#plt.close('all')



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

#Serial implementation:
if 1:
	for isamp in range(Sampler.iter,Input.Nsamp):

		Sampler.step()
		loglik = 0.0

		while True:
			Sampler.step()
			if(TSE.hetmodel.rate_model.check_bounds(TSE.hetmodel.get_rate(Sampler.param_new))):
				break
		
		loglik = 0.0


		for irep in range(TSE.Nrep):
			T1 = TSE.TS[irep]
			

			new_rates = TSE.hetmodel.rate_model.vary_coeff(TSE.hetmodel.get_rate(Sampler.param_new))
			T1.Equation.update_coeff(new_rates)


			new_error = TSE.hetmodel.error_model.vary_coeff(TSE.hetmodel.get_error_rep(Sampler.param_new,irep))
			T1.Error_model.update_coeff([new_error])

			ll = T1.calculate_likelihood_with_particle_filter(1000)
			loglik += ll
		
		accept = Sampler.decide(loglik)
		if accept == 1:
			Sampler.save(loglik,Input)




