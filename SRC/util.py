
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats

def step_OU(x,dt,theta1,theta2,theta3):

	xnew = x + (theta1 - theta2*x)*dt + theta3*np.random.normal()*np.sqrt(dt)

	return xnew 

def step_OU_dist(x,dt,theta1,theta2,theta3):

	mean1 = theta1/theta2 + (x - theta1/theta2)*np.exp(-theta2*dt)
	var1 = theta3*theta3*(1-np.exp(-2*theta2*dt))/(2*theta2)
	xnew = np.random.normal(loc=mean1,scale=np.sqrt(var1))

	return xnew 

def lik_OU(xat,x,dt,theta):

	theta1 = theta[0]; theta2 = theta[1]; theta3 = theta[2];

	mean1 = theta1/theta2 + (x - theta1/theta2)*np.exp(-theta2*dt)
	var1 = theta3*theta3*(1-np.exp(-2*theta2*dt))/(2*theta2)

	L = sp.stats.norm.pdf(xat,loc=mean1,scale=np.sqrt(var1))


	return L;


def loglik_OU(xat,x,dt,theta):

#Calculates the exact transition density from the analytical solution

	theta1 = theta[0]; theta2 = theta[1]; theta3 = theta[2];

	mean1 = theta1/theta2 + (x - theta1/theta2)*np.exp(-theta2*dt)
	var1 = theta3*theta3*(1-np.exp(-2*theta2*dt))/(2*theta2)

	L = sp.stats.norm.pdf(xat,loc=mean1,scale=np.sqrt(var1))


	return np.log(L);

def loglik_OU_particle(xat,x,dt,Nd,theta,Npart):

#Calculate the log-likelihood using particles
#This is very hard to do!!!
#Inference becomes much easier with measurement error

	theta1 = theta[0]; theta2 = theta[1]; theta3 = theta[2];
	

	mean1 = theta1/theta2 + (x - theta1/theta2)*np.exp(-theta2*dt)
	var1 = theta3*theta3*(1-np.exp(-2*theta2*dt))/(2*theta2)
	xex= np.random.normal(loc = np.repeat(mean1,10000),scale = np.sqrt(var1))


	count = 0;

	xnp1 = np.empty(Npart);
	dt = dt/Nd
	for ipart in range(Npart):
		xn = x
		for itime in range(Nd):
			xn = (xn + (theta1 - theta2*xn)*dt + theta3*np.sqrt(dt)*np.random.normal())
		xnp1[ipart] = xn

	if(0):
		print(dt)
		print(xnp1)
		plt.figure()
		sns.kdeplot(xex,label="exact")
		sns.kdeplot(xnp1,label="app")
		plt.axvline(x=x,color='green')
		plt.axvline(x=xat,color='red')
		plt.show()

	


	return 0#np.log(count/Npart);




