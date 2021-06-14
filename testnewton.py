from cqRK import CQModel
class ScatModel(CQModel):
	def precomputing(self,s):
		return s**2
	def harmonicForward(self,s,b,precomp = None):
		return precomp*b
	def harmonicBackward(self,s,b,precomp = None):
		return precomp**(-1)*b
	def righthandside(self,t,history = None):
		return 8*7*t**6
		#return 3*t**2+t**9
	def nonlinearity(self,x):
		return 0*x
		#return x**3
model = ScatModel()
T = 1
Am = 5
m=3
import numpy as np
taus = np.zeros(Am)
err = np.zeros(Am)
Ns  = np.zeros(Am)
comp = np.zeros(Am)
for j in range(Am):
	N = 8*2**j
	#N = 2*(2*2**j-1)
	Ns[j] = N
	tau = T*1.0/N
	taus[j] =tau 
	tt = np.linspace(0,T,N+1)
	#ex_sol = 1.0/30*tt**6
	ex_sol = tt**8
	#ex_sol = 4*3*tt**2
	#method = "BDF-"+str(m)
	method = "RadauIIA-"+str(m)
	sol = model.simulate(T,N,1,method = method)
	#err[j] = np.abs(sol[0,2]-tau**3)
	#print(err)
	err[j] = max(np.abs(sol[0,::m]-ex_sol))
	#err[j] = np.abs(sol[0,-1])

#	import matplotlib.pyplot as plt
#	print(sol)
#	plt.plot(sol[0,:])
#	plt.plot(ex_sol,linestyle='dashed')
#	plt.show()
print(err)
#print("NUMERICAL SOLUTION;")
#print(sol)
##print(sol[::2])
#print("EXACT SOLUTION:")
#print(ex_sol)
#print("ENDING ",sol)
#print(taus**3)
#import matplotlib.pyplot as plt
#plt.plot(sol[0,::2])
##plt.plot(ex_sol,linestyle='dashed')
#plt.show()

import matplotlib.pyplot as plt
plt.loglog(taus,err)
plt.loglog(taus,taus**1,linestyle = 'dashed')
plt.show()
	#
