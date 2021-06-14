from cq2 import CQModel2
class ScatModel(CQModel2):
	def precomputing(self,s):
		return s**1
	def harmonicForward(self,s,b,precomp = None):
		return precomp*b
	def harmonicBackward(self,s,b,precomp = None):
		return precomp**(-1)*b
	def righthandside(self,t,history = None):
		return 3*t**2
		#return 3*t**2+t**9
	def nonlinearity(self,x):
		return 0*x
		#return x**3
model = ScatModel()
T = 1
Am = 5
import numpy as np
taus = np.zeros(Am)
err = np.zeros(Am)
Ns  = np.zeros(Am)
comp = np.zeros(Am)
for j in range(Am):
	N = 7*2**j
	Ns[j] = N
	taus[j] = T*1.0/N
	tt = np.linspace(0,T,N+1)
	#ex_sol = 1.0/30*tt**6
	ex_sol = tt**3
	#ex_sol = 4*3*tt**2
	sol = model.simulate(T,N,1)
	err[j] = max(np.abs(sol[0,:]-ex_sol))
#	import matplotlib.pyplot as plt
#	print(sol)
#	plt.plot(sol[0,:])
#	plt.plot(ex_sol,linestyle='dashed')
#	plt.show()

import matplotlib.pyplot as plt
plt.loglog(taus,err)
plt.loglog(taus,taus**2,linestyle = 'dashed')
plt.show()
	
