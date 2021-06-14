#from CQ_acoustic import *
#import bempp.api

class ConvOperatoralt:
	import numpy as np
	tol=10**-14
	
	def __init__(self,apply_elliptic_operator,order=2):
		self.order=order
		self.delta=lambda zeta : self.char_functions(zeta,order)
		self.apply_elliptic_operator=apply_elliptic_operator
	def get_integration_parameters(self):
		N=self.N
		T=self.T
		tol=self.tol
		dt=(T*1.0)/N
		L=2*N
		rho=tol**(1.0/(1*L))
		return L,dt,tol,rho

	def char_functions(self,zeta,order):
		if order==1:
			return 1-zeta
		else:
			if order==2:
				return 1.5-2.0*zeta+0.5*zeta**2
			else:
				if order ==3:
					#return 1-zeta**3
					return (1-zeta)+0.5*(1-zeta)**2+1.0/3.0*(1-zeta)**3
				else:
					print("Multistep order not availible")

	def get_zeta_vect(self):
		L,dt,tol,rho=self.get_integration_parameters()
		import numpy as np
		#Calculating the Unit Roots
		Unit_Roots=np.exp(-1j*2*np.pi*(np.linspace(0,L,L+1)/(L+1)))
		#Calculating zetavect
		#Zeta_vect=map(lambda y: self.delta( rho* y)/dt , Unit_Roots)
		Zeta_vect=self.delta( rho* Unit_Roots)/dt 
		return Zeta_vect

	def apply_convol(self,rhs,T,show_progress=True,cutoff=10**(-8)):
		import numpy as np
		import bempp.api
		## Step 1
		print("In apply_convol")
		try:
			N=len(rhs[0,:])-1
		except:
			N=len(rhs)-1	
			rhs_mat=np.zeros((1,N+1))
			rhs_mat[0,:]=rhs
			rhs=rhs_mat
		self.N=N
		self.T=T
		rhs_fft=self.scale_fft(rhs)


		#import matplotlib.pyplot as plt
		#normsRHS=np.zeros(N)
		#for j in range(N):
		#	normsRHS[j]=np.linalg.norm(rhs[:,j])
			
		#plt.semilogy(normsRHS)
#		for j in range(0,len(rhs[:,0])):
#			print("rhs_fft : " , np.abs(rhs[j,:]))
#			plt.loglog(list(map(max,np.abs(rhs_fft[j,:]),10**-15*np.ones(len(rhs_fft[j,:])))))
#	

		#Initialising important parameters for the later stage	
		Zeta_vect=self.get_zeta_vect()

		dof=len(rhs[:,0])
		L,dt,tol,rho=self.get_integration_parameters()
		Half=int(np.ceil(float(L)/2.0))
		dummy = rhs
		## Step 2
		#phi_hat=1j*np.zeros((dof,L+1))
		normsRHS=np.ones(L+1)
		counter=0
		for j in range(0,Half+1):
			normsRHS[j]=np.max(np.abs(rhs_fft[:,j]))
			if normsRHS[j]>cutoff:
				counter=counter+1
		print("CUTOFF :", cutoff)
		
		#import matplotlib.pyplot as plt
		#plt.semilogy(normsRHS)
		print("Amount of Systems needed: "+ str(counter))
		#Timing the elliptic systems
		import time
		start=0
		end=0
		for j in range(0,Half+1):
			normsRHS[j]=np.max(np.abs(rhs_fft[:,j]))
			#print("normRHS:",normsRHS[j])
			if normsRHS[j]>cutoff:
				if show_progress:
					print("j:",j,"L:",str(Half), "Time of previous iteration: " +str((end-start)/60), " MIN" )
				start=time.time()
				if j>0:
					phi_hat[:,j]=self.apply_elliptic_operator(Zeta_vect[j],rhs_fft[:,j])
				else:
					first_eval=self.apply_elliptic_operator(Zeta_vect[j],rhs_fft[:,j])
					try:
						phi_hat=1j*np.zeros((len(first_eval),L+1))
						phi_hat[:,0]=first_eval
					except:
						phi_hat=1j*np.zeros((1,L+1))
						phi_hat[:,0]=first_eval
				end=time.time()
		for j in range(Half+1,L+1):
			phi_hat[:,j]=np.conj(phi_hat[:,L+1-j])		
		print(phi_hat)
		## Step 3
		phi_sol=self.rescale_ifft(phi_hat)
		if len(phi_sol[:,0])==1:
			phi_sol=phi_sol[0,:]
		print(phi_sol)
		return phi_sol



	def scale_fft(self,A):
		import numpy as np
		L,dt,tol,rho=self.get_integration_parameters()
		N=self.N
		n_rows=len(A[:,0])
		A_hat=1j*np.ones((n_rows,L+1))
		A_fft=1j*np.ones((n_rows,L+1))
		
		for j in range(0,n_rows):
			A_hat[j,:]=np.concatenate((rho**(np.linspace(0,N,N+1))*A[j,:],np.zeros(L-N)),axis=0)
			A_fft[j,:]=np.fft.fft(A_hat[j,:])
		print(len(A_hat[0]))
		return(A_fft)

	def rescale_ifft(self,A):
		import numpy as np
		L,dt,tol,rho=self.get_integration_parameters()
		N=self.N
		n_rows=len(A[:,0])
		ift_A=1j*np.ones((n_rows,L+1))
		A_sol=np.zeros((n_rows,N+1))
		for j in range(0,n_rows):
			ift_A[j,:]=np.fft.ifft(A[j,:])
			A_sol[j,:]=np.real(rho**(-np.linspace(0,N,N+1))*ift_A[j,0:N+1])
		return(A_sol)

