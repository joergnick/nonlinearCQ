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
        ## Methods supplied by user:
    def nonlinearity(self,x):
        raise NotImplementedError("No nonlinearity given.")
    def nonlinearityInverse(self,x):
        raise NotImplementedError("No inverse to nonlinearity given.")
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
    def newtonsolver(self,s,rhs,W0,Tdiag, x0,charMatrix0,tol = 10**(-8),coeff = 1,grada = None):
        dof = len(rhs)
        m = len(W0)
        for stageInd in range(m):
            for j in range(dof):
                if np.abs(x0[j,stageInd])<10**(-5):
                    x0[j,stageInd] = 10**(-5)
        Tinv = np.linalg.inv(Tdiag)
        rhsLong = 1j*np.zeros(m*dof)
        if grada is None:
             grada = np.zeros((m*dof,m*dof))
             taugrad = 10**(-8)
             idMat = np.identity(dof)
             for stageInd in range(m):
                 for i in range(dof):
                     diff = (self.nonlinearity(x0[:,stageInd]+taugrad*idMat[:,i])-self.nonlinearity(x0[:,stageInd]-taugrad*idMat[:,i]))
                     #if dof == 1:
                     grada[stageInd*dof:(stageInd+1)*dof,stageInd*dof+i] = diff/(2*taugrad)
        stageRHS = x0+1j*np.zeros((dof,m))
        ## Calculating right-hand side
        stageRHS = np.matmul(stageRHS,Tinv.T)
        for stageInd in range(m):
            stageRHS[:,stageInd] = self.harmonicForward(s,stageRHS[:,stageInd],precomp=W0[stageInd])
        stageRHS = np.matmul(stageRHS,Tdiag.T)
        rhsNewton = stageRHS+self.nonlinearity(x0)-rhs
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
            for j in range(m):  
                ydummy[j*dof:(j+1)*dof] = self.harmonicForward(s[j],xdummy[j*dof:(j+1)*dof],precomp = W0[j])
            ydummy = ydummy+np.matmul(np.matmul(Tdiagdof,grada),Tinvdof).dot(xdummy)
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
        if coeff*np.linalg.norm(dx)<tol:
            info = 0
        else:
            info = coeff*np.linalg.norm(dx)
        return np.real(x1),grada,info

    def fixedPointSolver(self,s,rhs,W0,Tdiag, x0,charMatrix0,tol = 10**(-8),coeff = 1):
        dof = len(rhs)
        m = len(W0)
        Tinv = np.linalg.inv(Tdiag)
        rhsLong = 1j*np.zeros(m*dof)
        x1 = x0
        counter = 0
        while (np.linalg.norm(x1-x0)>10**(-5)) or (counter==0 ):
            counter = counter+1
            if counter >= 20:
                break
            x0 = x1
            stageRHS = x0+1j*np.zeros((dof,m))
            ## Calculating right-hand side
            stageRHS = np.matmul(stageRHS,Tinv.T)
            for stageInd in range(m):
                stageRHS[:,stageInd] = self.harmonicForward(s,stageRHS[:,stageInd],precomp=W0[stageInd])
            stageRHS = np.matmul(stageRHS,Tdiag.T)
            rhsFP = rhs-self.nonlinearity(x0)
            rhsFP = np.matmul(rhsFP,Tinv.T)
            if np.isnan(rhsFP).any():
                raise ValueError("Nan value occurs in right-hand side")
            for stageInd in range(m):
                rhsLong[stageInd*dof:(stageInd+1)*dof] = rhsFP[:,stageInd]
            def FixPointFunc(xdummy):
                ydummy = 1j*np.zeros(dof*m)
                for j in range(m):  
                    ydummy[j*dof:(j+1)*dof] = self.harmonicForward(s[j],xdummy[j*dof:(j+1)*dof],precomp = W0[j])
                return ydummy
            FPLambda = lambda x: FixPointFunc(x)
            from scipy.sparse.linalg import LinearOperator
            FPOperator = LinearOperator((m*dof,m*dof),FPLambda)
            x1long,info = gmres(FPOperator,rhsLong,tol=1e-8)
            if info != 0:
                print("GMRES Info not zero, Info: ", info)
                ## Calculating Matrix
                FPMat = np.zeros((m*dof,m*dof))
                Mid = np.identity(m*dof)
                for k in range(m*dof):
                    FPMat[:,k] = FixPointFunc(Mid[:,k])
                print("Corresponding Matrix: ", FPMat, " Condition: ", np.linalg.cond(FPMat), " RHS : ",rhsLong)
            x1 = 1j*np.zeros((dof,m))
            for stageInd in range(m):
                x1[:,stageInd] = x1long[dof*stageInd:dof*(stageInd+1)]
            x1 = np.matmul(x1,Tdiag.T)  
        if (np.linalg.norm(x1-x0)<=10**(-5)):
            info = 0
        else:
            info = np.linalg.norm(x1-x0)
        return np.real(x1),info

    def reversefixedPointSolver(self,s,rhs,W0,Tdiag, x0,charMatrix0,tol = 10**(-8),coeff = 1):
        dof = len(rhs)
        m = len(W0)
        Tinv = np.linalg.inv(Tdiag)
        rhsLong = 1j*np.zeros(m*dof)
        x1 = x0
        counter = 0
        #print("NEW SYSTEM!")
        stageRHS = x0+1j*np.zeros((dof,m))
        x0 = x0
        while (np.linalg.norm(x1-x0)>10**(-8)) or (counter==0 ):
            counter = counter+1
            if counter >= 20:
                break
            x0 = x1
            ## Calculating right-hand side
            W0X0 = np.matmul(x0,Tinv.T)
            for stageInd in range(m):
                W0X0[:,stageInd] = self.harmonicForward(s,W0X0[:,stageInd],precomp=W0[stageInd])
            W0X0 = np.matmul(W0X0,Tdiag.T)
            rhsFP = rhs-W0X0
            x1 = 1j*np.zeros((dof,m))
            for stageInd in range(m):
                x1[:,stageInd] = self.nonlinearityInverse(rhsFP[:,stageInd])
            if np.linalg.norm(x1)>10**9:

                return np.real(x1),10**9
            #print("DISTANCE X1-X0 ",np.linalg.norm(x1-x0))
        if (np.linalg.norm(x1-x0)<=10**(-5)) and not np.isnan(x1).any():
            info = 0
        else:
            info = np.linalg.norm(x1-x0)
        return np.real(x1),info

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
            #print("NEW STEP : ",j, "ex_ sol: ",[(tj+c*tau)**3 for c in c_RK])
            for i in range(m):
                rhs[:,j*m+i+1] = rhs[:,j*m+i+1] + self.righthandside(tj+c_RK[i]*tau,history=sol[:,:j*m])
                if j >=1:
                    extr[:,i] = self.extrapol(sol[:,i+1:j*m+i+1:m],m+1)
                else:
                    extr[:,i] = np.zeros(dof)
            ############## SOLVING NONLINEAR SYSTEM########################
   #         ### Try 1: reverse fixed Point iteration:
   #         sol[:,j*m+1:(j+1)*m+1],info = self.reversefixedPointSolver(deltaEigs,rhs[:,j*m+1:(j+1)*m+1],W0,Tdiag,extr,charMatrix0)
   #         if info <=tolsolver:
   #             print("Used REVERSE Fix Point!")
   #         ### Try 2: forward fixed Point iteration:
   #        # if info >tolsolver:
   #        #     sol[:,j*m+1:(j+1)*m+1],info = self.fixedPointSolver(deltaEigs,rhs[:,j*m+1:(j+1)*m+1],W0,Tdiag,extr,charMatrix0)
   #        # if info >tolsolver:
   #        #     print("Both fixed point failed. info: "+str(info))
   #         ### Try 3: Classical Newton's method, 4 iterations
   #         sol[:,j*m+1:(j+1)*m+1] = extr
   #         for it in range(4):
   #             if info >tolsolver:
   #                 sol[:,j*m+1:(j+1)*m+1],info = self.newtonsolver(deltaEigs,rhs[:,j*m+1:(j+1)*m+1],W0,Tdiag,extr,charMatrix0)
   #         if info <=tolsolver:
   #             print("Used classical Newton!")

   #         ###  Use simplified Weighted Newon's method ######
            sol[:,j*m+1:(j+1)*m+1] = extr
            sol[:,j*m+1:(j+1)*m+1],grada,info = self.newtonsolver(deltaEigs,rhs[:,j*m+1:(j+1)*m+1],W0,Tdiag,sol[:,j*m+1:(j+1)*m+1],charMatrix0)
            counter = 0
            while info >0:
                    sol[:,j*m+1:(j+1)*m+1],grada,info = self.newtonsolver(deltaEigs,rhs[:,j*m+1:(j+1)*m+1],W0,Tdiag,sol[:,j*m+1:(j+1)*m+1],charMatrix0,grada=grada,coeff=0.5**counter)
                    counter = counter+1

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



