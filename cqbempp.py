from cqtoolbox import CQModel
from linearcq import Conv_Operator
from customOperators import precompMM,sparseWeightedMM,applyNonlinearity
import bempp.api
import os.path
import numpy as np
OrderQF = 8
bempp.api.global_parameters.quadrature.near.max_rel_dist = 2
bempp.api.global_parameters.quadrature.near.single_order =OrderQF-1
bempp.api.global_parameters.quadrature.near.double_order = OrderQF-1
bempp.api.global_parameters.quadrature.medium.max_rel_dist =4
bempp.api.global_parameters.quadrature.medium.single_order =OrderQF-2
bempp.api.global_parameters.quadrature.medium.double_order =OrderQF-2
bempp.api.global_parameters.quadrature.far.single_order =OrderQF-3
bempp.api.global_parameters.quadrature.far.double_order =OrderQF-3
bempp.api.global_parameters.quadrature.double_singular = OrderQF
bempp.api.global_parameters.hmat.eps=10**-5
bempp.api.global_parameters.hmat.admissibility='strong'
def a(x):
    return np.linalg.norm(x)**(-0.5)*x
#    return x
def Da(x):
#    if np.linalg.norm(x)<10**(-15):
#        x=10**(-15)*np.ones(3)
#    return np.eye(3)
    return -0.5*np.linalg.norm(x)**(-2.5)*np.outer(x,x)+np.linalg.norm(x)**(-0.5)*np.eye(3)
def calcRighthandside(c_RK,grid,N,T):
    m = len(c_RK)
    tau = T*1.0/N
    RT_space=bempp.api.function_space(grid, "RT",0)
    dof = RT_space.global_dof_count
    rhs = np.zeros((2*dof,m*N))
    curls = np.zeros((dof,m*N))
    for j in range(N):
        for stageInd in range(m):
            t = tau*j+tau*c_RK[stageInd] 
            def func_rhs(x,n,domain_index,result):
                inc =  np.array([np.exp(-50*(x[2]-t+2)**2), 0. * x[2], 0. * x[2]])    
                tang = np.cross(n,np.cross(inc, n))
                result[:] = tang
            tangtrace_inc = bempp.api.GridFunction(RT_space,fun = func_rhs,dual_space = RT_space)
            def func_curls(x,n,domain_index,result):
                curlU=np.array([ 0. * x[2],-100*(x[2]-t+2)*np.exp(-50*(x[2]-t+2)**2), 0. * x[2]])
                result[:] = np.cross(curlU,n)
            curlfun_inc = bempp.api.GridFunction(RT_space,fun = func_curls,dual_space = RT_space) 
            curls[:,j*m+stageInd]  = curlfun_inc.coefficients
            rhs[dof:,j*m+stageInd] = tangtrace_inc.coefficients
    def sinv(s,b):
        return s**(-1)*b
    IntegralOperator=Conv_Operator(sinv)
    gTH=IntegralOperator.apply_RKconvol(curls,T,method="RadauIIA-"+str(m),show_progress=False)
    gTH = np.concatenate((np.zeros((dof,1)),gTH),axis = 1)
    #rhs[0:dof,:]=np.real(gTH)-rhs[0:dof,:]
    return -gTH


#N =10
#dx = 1
#m=2
def nonlinearScattering(N,dx,m):
    grid = bempp.api.shapes.sphere(h=dx)
    RT_space=bempp.api.function_space(grid, "RT",0)
    gridfunList,neighborlist,domainDict = precompMM(RT_space)
    id_op=bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, RT_space)
    id_weak = id_op.weak_form()
    class ScatModel(CQModel):
        def precomputing(self,s):
            NC_space=bempp.api.function_space(grid, "NC",0)
            RT_space=bempp.api.function_space(grid, "RT",0)
            elec = -bempp.api.operators.boundary.maxwell.electric_field(RT_space, RT_space, NC_space,1j*s)
            magn = -bempp.api.operators.boundary.maxwell.magnetic_field(RT_space, RT_space, NC_space, 1j*s)
            identity= -bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, NC_space)
            identity2=bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, RT_space)
            blocks=np.array([[None,None], [None,None]])
            blocks[0,0] = -elec.weak_form()
            blocks[0,1] =  magn.weak_form()-1.0/2*identity.weak_form()
            blocks[1,0] = -magn.weak_form()-1.0/2*identity.weak_form()
            blocks[1,1] = -elec.weak_form()
            return [bempp.api.BlockedDiscreteOperator(blocks),identity2]
        def harmonicForward(self,s,b,precomp = None):
            return precomp[0]*b
        def calcJacobian(self,x,t,inhom):
            weightphiGF = bempp.api.GridFunction(RT_space,coefficients = x[:dof])
            weightIncGF = bempp.api.GridFunction(RT_space,coefficients = inhom)
            jacob = sparseWeightedMM(RT_space,weightphiGF+weightIncGF,Da,gridfunList,neighborlist,domainDict)
            return jacob
        def applyJacobian(self,jacob,x):
            dof = len(x)/2
            jx = 1j*np.zeros(2*dof)
            jx[:dof] = jacob*x[:dof]
            return jx
        def nonlinearity(self,coeff,t,inhom):
            dof = len(coeff)/2
            phiGridFun = bempp.api.GridFunction(RT_space,coefficients=coeff[:dof]) 
            gTHFun = bempp.api.GridFunction(RT_space,coefficients = inhom)
            agridFun= applyNonlinearity(phiGridFun+gTHFun,a,gridfunList,domainDict)
            result = np.zeros(2*dof) 
            result[:dof] = id_weak*agridFun.coefficients
            return result
    
        def righthandside(self,t,history=None):
            def func_rhs(x,n,domain_index,result):
                inc =  np.array([np.exp(-50*(x[2]-t+2)**2), 0. * x[2], 0. * x[2]])    
                tang = np.cross(n,np.cross(inc, n))
                curlU=np.array([ 0. * x[2],-100*(x[2]-t+2)*np.exp(-50*(x[2]-t+2)**2), 0. * x[2]])   
                result[:] = tang
                #return np.cross(curlU,n)
            RT_space=bempp.api.function_space(grid, "RT",0)
            gridfunrhs = bempp.api.GridFunction(RT_space,fun = func_rhs,dual_space = RT_space)
            dof = RT_space.global_dof_count
            rhs = np.zeros(dof*2)
            rhs[:dof] = gridfunrhs.coefficients
            rhs[:dof] = id_weak*gridfunrhs.coefficients
            #print(np.linalg.norm(rhs))
            return rhs
    model = ScatModel()
    import time
    start = time.time()
    [A_RK,b_RK,c_RK,m] = model.tdForward.get_method_characteristics("RadauIIA-"+str(m))
    T=4
    dof = RT_space.global_dof_count
    print("GLOBAL DOF: ",dof)
    rhsInhom = calcRighthandside(c_RK,grid,N,T)
    print("Finished RHS.")
    sol ,counters  = model.simulate(T,N,rhsInhom =rhsInhom, method = "RadauIIA-2")
    end = time.time()
    import matplotlib.pyplot as plt
    dof = RT_space.global_dof_count
    norms = [np.linalg.norm(sol[:,k]) for k in range(len(sol[0,:]))]
    return sol


AmTime = 7
AmSpace = 6
Ns = np.zeros(AmTime)
dxs = np.zeros(AmSpace)
m = 2
for indexSpace in range(AmSpace):
    for indexTime in range(AmTime):
        N  = 4*2**indexTime 
        Ns[indexTime] = N
        #dx = 0.75**(-indexSpace/2)
        dx = 2**(-indexSpace*1.0/2)
        filename = 'data/solh'+str(round(dx,3))+'N'+str(N)+'m'+str(m)+'.npy'
        if os.path.isfile(filename):
            print("The file "+filename+ " already exists, jumping simulation.")
            continue
        dxs[indexSpace] = dx

        print("Next file to be computed: "+filename)
        sol = nonlinearScattering(N,dx,m)
        np.save(filename,sol)

#np.save('data/counterssmall.npy',counters)
#plt.plot(norms)
#plt.show()
#norms = [bempp.api.GridFunction(RT_space,coefficients = sol[:dof,k]).l2_norm() for k in range(len(sol[0,:]))]
#plt.plot(norms)
#plt.show()
#gridfunphi = bempp.api.GridFunction(RT_space,coefficients = sol[:dof,-1])
#
##gridfunpsi = bempp.api.GridFunction(RT_space,coefficients = sol[.1,dof:])
#gridfunphi.plot()
#def harmonic_calderon(s,b,grid):
#        points=np.array([[0],[0],[2]])
#        #normb=np.linalg.norm(b[0])+np.linalg.norm(b[1])+np.linalg.norm(b[2])
#        normb=np.max(np.abs(b))
#        bound=np.abs(s)**4*np.exp(-s.real)*normb
#        print("s: ",s, " maxb: ", normb, " bound : ", bound)
#        if bound <10**(-9):
#                print("JUMPED")
#                return np.zeros(3)
#        
#        #tol= np.finfo(float).eps
#
####    Define Spaces
##       A_mat=bempp.api.as_matrix(blocks)
##       print("A_mat : ",A_mat)
##       e,D=np.linalg.eig(A_mat)
##       print("Eigs : ", e)
##       print("Cond : ", np.linalg.cond(A_mat))
###
###      trace_fun= bempp.api.GridFunction(multitrace.range_spaces[0], coefficients=b[0:dof],dual_space=multitrace.dual_to_range_spaces[0])
###
###      zero_fun= bempp.api.GridFunction(multitrace.range_spaces[1],coefficients = b[dof:],dual_space=multitrace.dual_to_range_spaces[1])
###
###      rhs=[trace_fun,zero_fun]
###
###      #print("Still living")
###      
#        #from bempp.api.linalg import gmres 
#        from scipy.sparse.linalg import gmres
#        print("Start GMRES : ")
##       def print_res(rk):
##               print("Norm of residual: "+ str(np.linalg.norm(rk)))
#        #print(np.linalg.norm(lambda_data))
#        #lambda_data,info = gmres(blocks, b,tol=10**-4,restart=50,maxiter=100,callback=print_res)
#        lambda_data,info = gmres(blocks, b,tol=10**-5,maxiter=300)
#        print("INFO :", info)
#        #lambda_data,info = gmres(blocks, b,tol=10**-4,callback=print_res)
#        #print("I survived!")
#        #from bempp.api.linalg import lu
#        #lambda_data = lu(elec, trace_fun)
#        #lambda_data.plot()
#        #print("Norm lambda_data : ",np.linalg.norm(lambda_data))
#        #if (np.linalg.norm(lambda_data)<10**-10):
#        phigrid=bempp.api.GridFunction(RT_space,coefficients=lambda_data[0:dof],dual_space=RT_space)
#        psigrid=bempp.api.GridFunction(RT_space,coefficients=lambda_data[dof:2*dof],dual_space=RT_space)
#
#        slp_pot = bempp.api.operators.potential.maxwell.electric_field(RT_space, points, s*1j)
#
#        dlp_pot = bempp.api.operators.potential.maxwell.magnetic_field(RT_space, points, s*1j)
#        print("Evaluate field : ")      
#        scattered_field_data = -slp_pot * phigrid+dlp_pot*psigrid
##       scattered_field_H = -slp_pot * psigrid-dlp_pot*phigrid
##       H = scattered_field_H.reshape(3,1)[:,0]
##       print(" E : ", E, " H : ", H)
##       print("Angle: ", np.dot(E,H), " Scalar product with conjugation : ", np.dot(np.conj(E),H))
##       print("NORM COMBINED OPERATOR :" , np.linalg.norm(scattered_field_data)/np.linalg.norm(b))
##       print(scattered_field_data)
##       print("NORM ScatteredField :", np.linalg.norm(scattered_field_data))
##
##       print("s : ", s)
##       print("NORM B :" ,np.linalg.norm(b))
#        if np.isnan(scattered_field_data).any():
#                print("NAN Warning, s = ", s)
#                scattered_field_data=np.zeros(np.shape(scattered_field_data)3
#        return scattered_field_data.reshape(3,1)[:,0]
