from linearcq import Conv_Operator
import numpy as np
from scipy.sparse.linalg import gmres
class CQModel2:
	def __init__(self):
		self.tdForward = Conv_Operator(self.forwardWrapper)
		self.freqObj = dict()
		self.freqUse = dict()
		self.countEv = 0
        ## Methods supplied by user:
	def nonlinearity(self,x):
		raise NotImplementedError("No nonlinearity given.")
	def harmonicForward(self,s,b,precomp = None):
		raise NotImplementedError("No time-harmonic forward operator given.")
	def harmonicBackward(self,s,b,precomp = None):
		raise NotImplementedError("No time-harmonic backward operator given.")
	def righthandside(self,t,history=None):
		raise NotImplementedError("No right-hand side given.")

        ## Optional method supplied by user:
	def precomputing(self,s):
		raise NotImplementedError("No precomputing given.")
	def preconditioning(self,precomp):
		raise NotImplementedError("No preconditioner given.")
        ## Methods provided by class
	def forwardWrapper(self,s,b):
		if s in self.freqObj:
			self.freqUse[s] = self.freqUse[s]+1
		else:
			self.freqObj[s] = self.precomputing(s)
			self.freqUse[s] = 1
		return self.harmonicForward(s,b,precomp=self.freqObj[s])

	def newtonsolver(self,s,rhs,W0, x0):
		dof = len(rhs)
		grada = np.zeros((dof,dof))
		taugrad = 10**(-8)
		### WARNING: WRONG GRADIENT
		print("WARNING; THE GRADIENT IS WRONG AND SHOULD BE UPDATED!")
		for j in range(dof):
			grada[j,j] = (self.nonlinearity(x0[j]+taugrad)-self.nonlinearity(x0[j]+taugrad))/(2*taugrad)
		rhs = W0*x0+self.nonlinearity(x0)-rhs
		dx,info = gmres(W0+grada,rhs,x0=x0)
		if info != 0:
			print("GMRES Info not zero, Info: ", info)
		x1 = x0-dx
		return x1

	def createFFTLengths(self,N):
		lengths = [1]
		it = 1
		while len(lengths)<=N:
			lengths.append( 2**it)
			lengths.extend(lengths[:-1][::-1])
			it = it+1
		return lengths

	def extrapolCoefficients(self,p):
	        coeffs = np.ones(p+1)
	        for j in range(p+1):
	                for m in range(p+1):
	                        if m != j:
	                                coeffs[j]=coeffs[j]*(p+1-m)*1.0/(j-m)
	        return coeffs
	def extrapol(self,u,p):
		if len(u[0,:])<=p+1:
			u = np.concatenate((np.zeros((len(u[:,0]),p+1-len(u[0,:]))),u),axis=1)
		extrU = np.zeros(len(u[:,0]))
		gammas = self.extrapolCoefficients(p)
		for j in range(p+1):
			extrU = extrU+gammas[j]*u[:,-p-1+j]
		return extrU
	
	def simulate(self,T,N,dof):
		tau = T*1.0/N
		## Initializing right-hand side:
		lengths = self.createFFTLengths(N)
		## Actual solving:
		zeta0 = self.tdForward.delta(0)/tau
		W0 = self.precomputing(zeta0)
		rhs = np.zeros((dof,N+1))
		sol = np.zeros((dof,N+1))
		for j in range(1,N+1):
			## Calculating solution at timepoint tj
			tj       = tau*j
			rhs[:,j] = rhs[:,j] + self.righthandside(tj,history=sol[:,j-1])
			if j >=1:
				extr = self.extrapol(sol[:,:j],2)
				sol[:,j] = self.newtonsolver(zeta0,rhs[:,j],W0,extr)
			else:
				extr = self.extrapol(np.zeros((dof,1)),2)
				sol[:,j] = self.newtonsolver(zeta0,rhs[:,j],W0,extr)
			## Calculating Local History:
			currLen = lengths[j]
			localHist = np.concatenate((sol[:,j+1-currLen:j+1],np.zeros((dof,currLen))),axis=1)
			if len(localHist[0,:])>=1:
				localconvHist = self.tdForward.apply_convol(localHist,(len(localHist[0,:])-1)*tau)
			else:
				break
			## Updating Global History:	
			currLenCut = min(currLen,N-j)
			rhs[:,j+1:j+1+currLenCut] = rhs[:,j+1:j+1+currLenCut]-localconvHist[:,currLen:currLen+currLenCut]
		self.freqUse = dict()
		self.freqObj = dict()
		return sol 



