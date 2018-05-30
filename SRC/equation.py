import numpy as np
from scipy.stats import norm


#Define an equation class

class Equation:

	def __init__(self,name,coeff):
		self.name = name
		self.coeff = coeff
		self.Ndisc = 10

	def update_coeff(self,coeff):
		self.coeff = coeff

	def get_name(self):
		return(self.name)

	def update_Ndisc(self,Ndisc):
		self.Ndisc = Ndisc

#Define particular equation class from general equation class
class OU_Equation(Equation):

	def __init__(self,coeff):
		Equation.__init__(self,'OU',coeff)

	def get_dim(self):
		return 3;

	def simulate_exact(self,x,dt):
		coeff = self.coeff
		mean= coeff[0]/coeff[1] + (x - coeff[0]/coeff[1])*np.exp(-coeff[1]*dt)
		sd = coeff[2]*coeff[2]*( 1 - np.exp(-2*coeff[1]*dt))/(2*coeff[1])
		x = np.random.normal(loc=mean, scale=sd)
		return(x)

	def simulate_exact_v(self,x,dt):
		coeff = self.coeff
		mean= coeff[0]/coeff[1] + (x - coeff[0]/coeff[1])*np.exp(-coeff[1]*dt)
		sd = coeff[2]*coeff[2]*( 1 - np.exp(-2*coeff[1]*dt))/(2*coeff[1])
		x = np.random.normal(loc=mean, scale=sd)
		return(x)

	def calculate_transition_density(self,x,y,dt):
		coeff = self.coeff
		mean= coeff[0]/coeff[1] + (x - coeff[0]/coeff[1])*np.exp(-coeff[1]*dt)
		sd = coeff[2]*coeff[2]*( 1 - np.exp(-2*coeff[1]*dt))/(2*coeff[1])
		x = norm.pdf(y,loc=mean, scale=sd)
		return(x)


class BDI_Equation(Equation):

	def __init__(self,coeff):
		Equation.__init__(self,'BDI',coeff)

	def get_coeff_name(self):
		return (('kd','kf','gamma'))

	def get_dim(self):
		return 3;

	def simulate_exact(self,x,dt):#Gillespie algorithm
		coeff = self.coeff
		#print(coeff)
		#input()
		t = 0
		while(1):
			#print(x,t,dt)
			h1=coeff[0]; h2=coeff[1]*x; h3=coeff[2]*x;
			h0 = h1 + h2 + h3

			if h0<1.0e-10:
				t = 1e99

			t += np.random.exponential(scale=1.0/h0)

			if t > dt:
				break;

			u=np.random.uniform();

			if u<h1/h0:
				x+=1;
			elif u<(h1+h2)/h0:
				x+=1;
			else:
				x-=1;
		return(x)  

	def simulate_approx(self,x,dt):#Approx algorithm
		coeff = self.coeff
		dt_Euler = dt/self.Ndisc
		for i in range(self.Ndisc): 

			mean= x+dt_Euler*(coeff[0] + (-coeff[2]+coeff[1])*x)
			sd = np.sqrt(dt_Euler*(coeff[0] + (coeff[2]+coeff[1])*x))
			x = np.random.normal(loc=mean, scale=sd)
		

		return(x)

	def simulate_exact_v(self,x,dt):#Gillespie algorithm in vector form
		coeff = self.coeff

		xout = np.empty(x.size)

		for ipart in range(x.size):
			xp = x[ipart]
			t = 0
			while(1):
				h1=coeff[0]; h2=coeff[1]*xp; h3=coeff[2]*xp;
				h0 = h1 + h2 + h3

				if h0<1.0e-10:
					t = 1e99

				t += np.random.exponential(scale=1.0/h0)

				if t > dt:
					break;

				u=np.random.uniform();

				if u<h1/h0:
					xp+=1;
				elif u<(h1+h2)/h0:
					xp+=1;
				else:
					xp-=1;
			xout[ipart] = xp
		return(xout)  




