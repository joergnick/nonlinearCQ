from linearcq import Conv_Operator
import numpy as np
class CQModel:
	## Methods supplied by user:
	def nonlinearity(self,x):
		raise NotImplementedError("No nonlinearity given.")
	def harmonicForward(self,s,b,precomp = None):
		return s**(2)*b
		#raise NotImplementedError("No time-harmonic forward operator given.")
	def harmonicBackward(self,s,b,precomp = None):
		return s**(-2)*b
		#raise NotImplementedError("No time-harmonic backward operator given.")
	def righthandside(self,t,history):
		return t**4
	#	raise NotImplementedError("No right-hand side given.")

	## Optional method supplied by user:
	def precomputing(self,s):
		raise NotImplementedError("No precomputing given.")

	## Methods provided by class
	def newtonsolver(self,s,rhs, x0 = 0):
		return self.harmonicBackward(s,rhs)

	def simulate(self,T,N,dof):
		sol = np.zeros((dof,N+1))
		TDForward = Conv_Operator(self.harmonicForward,order=2)
		zetavect = TDForward.get_zeta_vect(N,T)
		tau = T*1.0/N
		zeta0 = TDForward.delta(0)/tau
		for j in range(1,N+1):
			tj = tau*j
			history = np.concatenate((sol[:,:j],np.zeros((dof,1))),axis = 1)
			history = np.concatenate((np.zeros((dof,N-j)),history),axis = 1)
			print("ITERATION:")
			print(len(history[0,:]))
			print(N+1)
			convhistory = TDForward.apply_convol(history,T) 
			tdRHS = self.righthandside(tj,history)-convhistory[:,N]
			sol[:,j] = self.newtonsolver(zeta0,tdRHS)
		return sol
			
		

	
