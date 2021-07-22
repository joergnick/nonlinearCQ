from scipy.sparse.linalg import gmres
from scipy.optimize import newton_krylov
import numpy as np
from linearcq import Conv_Operator
class CQModel:
    def __init__(self):
        self.tdForward = Conv_Operator(self.forwardWrapper)
        self.freqObj = dict()
        self.freqUse = dict()
        self.countEv = 0
    def time_step(self,s0,t,history,conv_history,x0):
        raise NotImplementedError("No time stepping given.") 
        ## Methods supplied by user:
    def nonlinearity(self,x):
        raise NotImplementedError("No nonlinearity given.")
    def nonlinearityInverse(self,x):
        raise NotImplementedError("No inverse to nonlinearity given.")
    def harmonicForward(self,s,b,precomp = None):
        raise NotImplementedError("No time-harmonic forward operator given.")
    def harmonicBackward(self,s,b,precomp = None):
        raise NotImplementedError("No time-harmonic backward operator given.")
    def calcGradient(self,x):
        raise NotImplementedError("No gradient given.")
    def applyGradient(self,b,grad=None):
        raise NotImplementedError("Gradient has no applyGradient method.") 
    def righthandside(self,t,history=None):
        return 0
        #raise NotImplementedError("No right-hand side given.")

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
    def discreteGradient(self,m,dof,x0):
        taugrad = 10**(-8)
        idMat = np.identity(dof)
        gradList = m*[None]
        for stageInd in range(m):
            grada = np.zeros((dof,dof))
            for i in range(dof):
                diff = (self.nonlinearity(x0[:,stageInd]+taugrad*idMat[:,i])-self.nonlinearity(x0[:,stageInd]-taugrad*idMat[:,i]))
                #if dof == 1:
                grada[:,i] = diff/(2*taugrad)
                #grada[stageInd*dof:(stageInd+1)*dof,stageInd*dof+i] = diff/(2*taugrad)
            gradList[stageInd] = grada 
        return gradList

    def newtonsolver(self,s,rhs,W0,Tdiag, x0,charMatrix0,tol = 10**(-8),gradList =None,coeff = 1):
        dof = len(rhs)
        m = len(W0)
        for stageInd in range(m):
            for j in range(dof):
                if np.abs(x0[j,stageInd])<10**(-5):
                    x0[j,stageInd] = 10**(-5)
        Tinv = np.linalg.inv(Tdiag)
        rhsLong = 1j*np.zeros(m*dof)
        print("x0 Shape",x0.shape)
            try:
                gradList = [self.calcGradient(x0[:,k]) for k in range(m)]
            except NotImplementedError:
                gradList = self.discreteGradient(m,dof,x0)
        stageRHS = x0+1j*np.zeros((dof,m))
        ## Calculating right-hand side

        stageRHS = np.matmul(stageRHS,Tinv.T)
        for stageInd in range(m):
            stageRHS[:,stageInd] = self.harmonicForward(s,stageRHS[:,stageInd],precomp=W0[stageInd])
        stageRHS = np.matmul(stageRHS,Tdiag.T)
        #rhsNewton = [stageRHS[:,k]+self.nonlinearity(x0[:,k])-rhs[:,k] for k in range(m)]
        ax0 = np.zeros((dof,m))
        for stageInd in range(m):
            ax0[:,stageInd] = self.nonlinearity(x0[:,stageInd])

        
        rhsNewton = stageRHS+ax0-rhs
        ## Solving system W0y = b
        #print("SecondMatrix: ",np.matmul(np.matmul(Tdiag,grada),Tinv))
        rhsNewton = np.matmul(rhsNewton,Tinv.T)
        #print("rhsNewton",rhsNewton)
        for stageInd in range(m):
            rhsLong[stageInd*dof:(stageInd+1)*dof] = rhsNewton[:,stageInd]
                        
        def NewtonFunc(xdummy):
            idMat  = np.identity(dof)
            Tinvdof = np.kron(Tinv,idMat)
            Tdiagdof = np.kron(Tdiag,idMat)
            ydummy = 1j*np.zeros(dof*m)
            Txdummy = Tinvdof.dot(xdummy)
            for j in range(m):  
                ydummy[j*dof:(j+1)*dof] = self.harmonicForward(s[j],xdummy[j*dof:(j+1)*dof],precomp = W0[j])
                Txdummy[j*dof:(j+1)*dof] =gradList[j].dot(Txdummy[j*dof:(j+1)*dof])
                #Txdummy[j*dof:(j+1)*dof] =grada[j*dof:(j+1)*dof,j*dof:(j+1)*dof].dot(Txdummy[j*dof:(j+1)*dof])
            ydummy = ydummy+Tdiagdof.dot(Txdummy)
            return ydummy
        NewtonLambda = lambda x: NewtonFunc(x)
        from scipy.sparse.linalg import LinearOperator
        NewtonOperator = LinearOperator((m*dof,m*dof),NewtonLambda)
        dxlong,info = gmres(NewtonOperator,rhsLong,tol=1e-8)
        if info != 0:
            print("GMRES Info not zero, Info: ", info)
            ## Calculating Matrix
            NewtonMat = np.zeros((m*dof,m*dof))
            Mid = np.identity(m*dof)
            for k in range(m*dof):
                NewtonMat[:,k] = NewtonFunc(Mid[:,k])
            print("Corresponding Matrix: ", NewtonMat, " Condition: ", np.linalg.cond(NewtonMat), " RHS : ",rhsLong)
        dx = 1j*np.zeros((dof,m))
        for stageInd in range(m):
            dx[:,stageInd] = dxlong[dof*stageInd:dof*(stageInd+1)]
        dx = np.matmul(dx,Tdiag.T)  
        x1 = x0-coeff*dx
        #print("RESIDUUM: ",np.linalg.norm(dx))
        if coeff*np.linalg.norm(dx)/dof<tol:
            info = 0
        else:
            info = coeff*np.linalg.norm(dx)
        return np.real(x1),gradList,info

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
    
    def simulate(self,T,N,method = "RadauIIA-2",tolsolver = 10**(-8)):
        tau = T*1.0/N
        ## Initializing right-hand side:
        lengths = self.createFFTLengths(N)
        try:
            dof = len(self.righthandside(0))
        except:
            dof = 1
        ## Actual solving:
        [A_RK,b_RK,c_RK,m] = self.tdForward.get_method_characteristics(method)
        charMatrix0 = np.linalg.inv(A_RK)/tau
        deltaEigs,Tdiag =np.linalg.eig(charMatrix0) 
    #   print(np.matmul(Tdiag,np.linalg.inv(Tdiag)))
    #   print(np.matmul(np.matmul(np.linalg.inv(Tdiag),charMatrix0),Tdiag))
        W0 = []
        for j in range(m):
            W0.append(self.precomputing(deltaEigs[j]))
        #zeta0 = self.tdForward.delta(0)/tau
        #W0 = self.precomputing(zeta0)
        rhs = np.zeros((dof,m*N+1))
        sol = np.zeros((dof,m*N+1))
        extr = np.zeros((dof,m))
        for j in range(0,N):
            ## Calculating solution at timepoint tj
            tj       = tau*j
            print("NEW STEP : ",j)
            for i in range(m):
                rhs[:,j*m+i+1] = rhs[:,j*m+i+1] + self.righthandside(tj+c_RK[i]*tau,history=sol[:,:j*m])
                if j >=1:
                    extr[:,i] = self.extrapol(sol[:,i+1:j*m+i+1:m],m)
                else:
                    extr[:,i] = np.zeros(dof)
   #         ###  Use simplified Weighted Newon's method ######
            sol[:,j*m+1:(j+1)*m+1] = extr
            sol[:,j*m+1:(j+1)*m+1],gradList,info = self.newtonsolver(deltaEigs,rhs[:,j*m+1:(j+1)*m+1],W0,Tdiag,sol[:,j*m+1:(j+1)*m+1],charMatrix0 = np.zeros((2*dof,2*dof)))
            print("Extrapolation-sol : ",np.linalg.norm(sol[:,j*m+1:(j+1)*m+1]-extr))
            counter = 0
            while info >0:
                    sol[:,j*m+1:(j+1)*m+1],grada,info = self.newtonsolver(deltaEigs,rhs[:,j*m+1:(j+1)*m+1],W0,Tdiag,sol[:,j*m+1:(j+1)*m+1],charMatrix0,gradList=gradList,coeff=0.5**counter)
                    counter = counter+1
            print(counter)

            ## Solving Completed #####################################
            ## Calculating Local History:
            currLen = lengths[j]
            localHist = np.concatenate((sol[:,m*(j+1)+1-m*currLen:m*(j+1)+1],np.zeros((dof,m*currLen))),axis=1)
            if len(localHist[0,:])>=1:
                localconvHist = np.real(self.tdForward.apply_RKconvol(localHist,(len(localHist[0,:]))*tau/m,method = method,show_progress=False))
            else:
                break
            ## Updating Global History: 
            currLenCut = min(currLen,N-j-1)
            rhs[:,(j+1)*m+1:(j+1)*m+1+currLenCut*m] = rhs[:,(j+1)*m+1:(j+1)*m+1+currLenCut*m]-localconvHist[:,currLen*m:currLen*m+currLenCut*m]
        self.freqUse = dict()
        self.freqObj = dict()
        return sol 
 
#    def integrate(self,T,N,method = "RadauIIA-2",tolsolver = 10**(-8)):
#        tau = T*1.0/N
#        ## Initializing right-hand side:
#        lengths = self.createFFTLengths(N)
#        try:
#            dof = len(self.righthandside(0))
#        except:
#            dof = 1
#        ## Actual solving:
#        [A_RK,b_RK,c_RK,m] = self.tdForward.get_method_characteristics(method)
#        charMatrix0 = np.linalg.inv(A_RK)/tau
#        deltaEigs,Tdiag =np.linalg.eig(charMatrix0) 
#    #   print(np.matmul(Tdiag,np.linalg.inv(Tdiag)))
#    #   print(np.matmul(np.matmul(np.linalg.inv(Tdiag),charMatrix0),Tdiag))
#        W0 = []
#        for j in range(m):
#            W0.append(self.precomputing(deltaEigs[j]))
#        #zeta0 = self.tdForward.delta(0)/tau
#        #W0 = self.precomputing(zeta0)
#        rhs = np.zeros((dof,m*N+1))
#        sol = np.zeros((dof,m*N+1))
#        extr = np.zeros((dof,m))
#        for j in range(0,N):
#            ## Calculating solution at timepoint tj
#            tj       = tau*j
#	    print(j)
#            #print("NEW STEP : ",j, "ex_ sol: ",[(tj+c*tau)**3 for c in c_RK])
#            for i in range(m):
#                rhs[:,j*m+i+1] = rhs[:,j*m+i+1] + self.righthandside(tj+c_RK[i]*tau,history=sol[:,:j*m])
#                if j >=1:
#                    extr[:,i] = self.extrapol(sol[:,i+1:j*m+i+1:m],m+1)
#                else:
#                    extr[:,i] = np.zeros(dof)
#   #         ###  Use simplified Weighted Newon's method ######
#            sol[:,j*m+1:(j+1)*m+1] = extr
#            sol[:,j*m+1:(j+1)*m+1] = self.time_step(W0,tj,,rhs[:,j*m+1:(j+1)*m+1],sol[:,j*m+1:(j+1)*m+1],Tdiag,charMatrix0)
#            #sol[:,j*m+1:(j+1)*m+1],grada,info = self.time_step(W0,rhs[:,j*m+1:(j+1)*m+1],W0,Tdiag,sol[:,j*m+1:(j+1)*m+1],charMatrix0)
#            ## Solving Completed #####################################
#            ## Calculating Local History:
#            currLen = lengths[j]
#            localHist = np.concatenate((sol[:,m*(j+1)+1-m*currLen:m*(j+1)+1],np.zeros((dof,m*currLen))),axis=1)
#            if len(localHist[0,:])>=1:
#                localconvHist = np.real(self.tdForward.apply_RKconvol(localHist,(len(localHist[0,:]))*tau/m,method = method,show_progress=False))
#            else:
#                break
#            ## Updating Global History: 
#            currLenCut = min(currLen,N-j-1)
#            rhs[:,(j+1)*m+1:(j+1)*m+1+currLenCut*m] = rhs[:,(j+1)*m+1:(j+1)*m+1+currLenCut*m]-localconvHist[:,currLen*m:currLen*m+currLenCut*m]
#        self.freqUse = dict()
#        self.freqObj = dict()
#        return sol 
#


