from cqRK import CQModel
import numpy as np
class ScatModel(CQModel):
	def precomputing(self,s):
		return s**(-1)
	def harmonicForward(self,s,b,precomp = None):
		return precomp*b
	def righthandside(self,t,history = None):
		return 1.0/4*t**4+t**1.5
	def nonlinearity(self,x):
	#	print("X IN NONLINEARITY: ",x," RESULT : ",abs(x)**(-0.5)*x)
		val = np.abs(x)**(-0.5)*x
		nanindizes = np.isnan(val)	
		val[nanindizes] = 0
		return val
		#return 0*x
		#return np.array([x[0]**1+x[1]**2,x[0]**3+x[1]**(1)])
	
model = ScatModel()
T = 1
Am = 1
m=3
taus = np.zeros(Am)
err = np.zeros(Am)
Ns  = np.zeros(Am)
comp = np.zeros(Am)
#print(np.kron(np.array([[1,2],[3,4]]),np.identity(3)))
for j in range(Am):
	N = 8*2**j
	#N = 2*(2*2**j-1)
	Ns[j] = N
	tau = T*1.0/N
	taus[j] =tau 
	tt = np.linspace(0,T,N+1)
	#ex_sol = 1.0/30*tt**6
	ex_sol = tt**3
	#ex_sol = 4*3*tt**2
	#method = "BDF-"+str(m)
	method = "RadauIIA-"+str(m)
	sol = model.simulate(T,N,1,method = method)

	#err[j] = np.abs(sol[0,2]-tau**3)
	#print(err)
	err[j] = max(np.abs(sol[0,::m]-ex_sol))
	#err[j] = np.abs(sol[0,-1])

	import matplotlib.pyplot as plt
#	print(sol)
	plt.plot(sol[0,::m])
	#plt.plot(sol[1,::m])
	plt.plot(ex_sol,linestyle='dashed')
	plt.show()
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
#
#import matplotlib.pyplot as plt
#plt.loglog(taus,err)
#plt.loglog(taus,taus**2,linestyle = 'dashed')
#plt.show()
	
