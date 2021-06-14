from scipy.sparse.linalg import gmres
import numpy as np
from linearcq import Conv_Operator
class CQModel:
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

	def newtonsolver(self,s,rhs,W0,Tdiag, x0,charMatrix0):
		dof = len(rhs)
		m = len(W0)
		Tinv = np.linalg.inv(Tdiag)
		grada = np.zeros((m*dof,m*dof))
		rhsLong = 1j*np.zeros(m*dof)
		taugrad = 10**(-8)
		for j in range(dof):
			for i in range(m):
				grada[i*dof+j,i*dof+j] = (self.nonlinearity(x0[j,i]+taugrad)-self.nonlinearity(x0[j,i]+taugrad))/(2*taugrad)
		#rhs = W0*x0+self.nonlinearity(x0)-rhs
		stageRHS = 1j*np.zeros((dof,m))
		## Calculating right-hand side
		stageRHS = np.matmul(stageRHS,Tinv.T)
		for stageInd in range(m):
			stageRHS[:,stageInd] = self.harmonicForward(s,stageRHS[:,stageInd],precomp=W0[stageInd])
		stageRHS = np.matmul(stageRHS,Tdiag.T)
		rhsNewton = stageRHS+self.nonlinearity(x0)-rhs
		## Solving system W0y = b
		rhsNewton = np.matmul(rhsNewton,Tinv.T)
		for stageInd in range(m):
			rhsLong[stageInd*dof:(stageInd+1)*dof] = rhsNewton[:,stageInd]
		def NewtonFunc(xdummy):
			ydummy = 1j*np.zeros(dof*m)
			for j in range(m):	
				ydummy[j*dof:(j+1)*dof] = self.harmonicForward(s[j],xdummy[j*dof:(j+1)*dof],precomp = W0[j])
			return ydummy
		from scipy.sparse.linalg import LinearOperator
		NewtonLambda = lambda x: NewtonFunc(x)
		NewtonOperator = LinearOperator((m*dof,m*dof),NewtonLambda)
		dxlong,info = gmres(NewtonOperator,rhsLong,tol=1e-8)
		if info != 0:
			print("GMRES Info not zero, Info: ", info)
		dx = 1j*np.zeros((dof,m))
		for stageInd in range(m):
			dx[:,stageInd] = dxlong[dof*stageInd:dof*(stageInd+1)]
		### WARNING TRANSPOSING MIGHT CAUSE TROUBLE WITH BEMPP
		dx = np.matmul(dx,Tdiag.T)	
		x1 = x0-dx
		return x1
		#return np.matmul(rhsNewton,np.linalg.inv(charMatrix0).T)

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
	
	def simulate(self,T,N,dof,method = "RadauIIA-2"):
		tau = T*1.0/N
		## Initializing right-hand side:
		lengths = self.createFFTLengths(N)
		## Actual solving:
		[A_RK,b_RK,c_RK,m] = self.tdForward.get_method_characteristics(method)
		charMatrix0 = np.linalg.inv(A_RK)/tau
		deltaEigs,Tdiag =np.linalg.eig(charMatrix0)	
		print("Used method: ", method)
		print(charMatrix0)
		print("Cond ", np.linalg.cond(Tdiag))
	#	print(np.matmul(Tdiag,np.linalg.inv(Tdiag)))
	#	print(np.matmul(np.matmul(np.linalg.inv(Tdiag),charMatrix0),Tdiag))
		W0 = []
		for j in range(m):
			W0.append(self.precomputing(deltaEigs[j]))
		#zeta0 = self.tdForward.delta(0)/tau
		#W0 = self.precomputing(zeta0)
		rhs = np.zeros((dof,m*N+1))
		sol = np.zeros((dof,m*N+1))
		for j in range(0,N):
			#print(j)
			## Calculating solution at timepoint tj
			tj       = tau*j
			for i in range(m):
				rhs[:,j*m+i+1] = rhs[:,j*m+i+1] + self.righthandside(tj+c_RK[i]*tau,history=sol[:,:j*m])
#			if j >=2:
#				extr = self.extrapol(sol[:,:j*m+i:m],2*m)
#				sol[:,j] = self.newtonsolver(zeta0,rhs[:,j],W0,1,extr)
#			else:
#				extr = np.zeros(dof)
#				sol[:,j] = self.newtonsolver(zeta0,rhs[:,j],W0,1,extr)
#
			extr = np.zeros((dof,m))
			sol[:,j*m+1:(j+1)*m+1] = np.real(self.newtonsolver(deltaEigs,rhs[:,j*m+1:(j+1)*m+1],W0,Tdiag,extr,charMatrix0))
			## Calculating Local History:
			currLen = lengths[j]
			localHist = np.concatenate((sol[:,m*(j+1)+1-m*currLen:m*(j+1)+1],np.zeros((dof,m*currLen))),axis=1)
			if len(localHist[0,:])>=1:
				localconvHist = np.real(self.tdForward.apply_RKconvol(localHist,(len(localHist[0,:]))*tau/m,method = method,show_progress=False))
			else:
				break
			## Updating Global History:	
			currLenCut = min(currLen,N-j-1)
		#	print(localconvHist)
		#	print(localconvHist[:,currLen*m:currLen*m+currLenCut*m])
		#	print(currLen)
		#	print(currLenCut)
			rhs[:,(j+1)*m+1:(j+1)*m+1+currLenCut*m] = rhs[:,(j+1)*m+1:(j+1)*m+1+currLenCut*m]-localconvHist[:,currLen*m:currLen*m+currLenCut*m]
		self.freqUse = dict()
		self.freqObj = dict()
		return sol 



