import numpy as np
import bempp.api
import inspect
import matplotlib.pyplot as plt

def squarify(rectMat):
	dof = len(rectMat[0,:])
	squareMat = np.zeros((dof,dof))
	index_sq_mat   = 0
	index_rect_mat = 0
	indizes = np.zeros(dof)
	from scipy.linalg import svdvals
	while (index_sq_mat < dof):
		squareMat[index_sq_mat,:] = rectMat[index_rect_mat,:]
		index_rect_mat = index_rect_mat+1
		#if (np.linalg.matrix_rank(squareMat[:index_sq_mat+1,:]) == (index_sq_mat+1)):
		if min(svdvals(squareMat[:index_sq_mat+1,:]))>10**(-3):
			indizes[index_sq_mat] = index_rect_mat
			index_sq_mat = index_sq_mat + 1	
	return squareMat,indizes

def getCoefftoEval(space):
	dof = space.global_dof_count
	trace_fun = bempp.api.GridFunction(space, coefficients = np.zeros(dof))
	centerdata = trace_fun.evaluate_on_element_centers()
	centerdataSize = len(centerdata[0,:])*3
	evalvertices = trace_fun.evaluate_on_vertices()
	verticedataSize = len(evalvertices[0,:])*3
	CoeffToEval = np.zeros((centerdataSize+verticedataSize,dof))
	coefficients = np.zeros(dof)
	for j in range(dof):
		coefficients[j] = 1
		trace_fun= bempp.api.GridFunction(space, coefficients = coefficients)
		evalcenters = trace_fun.evaluate_on_element_centers()
		CoeffToEval[:centerdataSize,j] = np.ravel(evalcenters)
		evalvertices = trace_fun.evaluate_on_vertices()
		CoeffToEval[centerdataSize:,j] = np.ravel(evalvertices)
		coefficients[j] = 0
	return CoeffToEval

def applyToGridFunction(gridfun,a,temp,CoeffToEval = None):
	if CoeffToEval is None:
		CoeffToEval = getCoefftoEval(gridfun.space)
	coeff = gridfun.coefficients
	centerEvalsGiven = gridfun.evaluate_on_element_centers()	
	verticeEvalsGiven = gridfun.evaluate_on_vertices()
	evals = CoeffToEval.dot(coeff)
	for j in range(len(evals)/3):
		evals[3*(j):3*(j+1)] = a(evals[3*j:3*(j+1)])
	import scipy
	coeffa,res,rank,s = scipy.linalg.lstsq(CoeffToEval,evals)
	print("Residuum: ",res)
	return bempp.api.GridFunction(RT_space,coefficients = coeffa)


def a(x):
	return x[0]*x
	#return np.linalg.norm(x)**(-0.5)*x
import inspect
print(inspect.getsource(bempp.api.GridFunction))
dx = 1
grid = bempp.api.shapes.sphere(h=dx)
RT_space = bempp.api.function_space(grid,"RT",0)
identity2=bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, RT_space)
import inspect
print(inspect.getsource(bempp.api.operators.boundary.sparse.identity))
#def tangential_trace(x, n, domain_index, result):
#        result[:] = np.cross(n,np.cross(np.array([np.exp(-1*(x[2])), 0. * x[2], 0. * x[2]]), n))
#def atang_trace(x,n,domain_index,result):
#	result[:] = a(np.cross(n,np.cross(np.array([np.exp(-1*(x[2])), 0. * x[2], 0. * x[2]]), n)))
#trace_fun  = bempp.api.GridFunction(RT_space, fun=tangential_trace,dual_space=RT_space)
#exatrace_fun  = bempp.api.GridFunction(RT_space, fun=atang_trace,dual_space=RT_space)
#atrace_fun = applyToGridFunction(trace_fun,a,exatrace_fun.coefficients)
#(exatrace_fun-atrace_fun).plot()
#print((exatrace_fun-atrace_fun).l2_norm())
##Am = 8
##errs = np.zeros(Am)
##l2errs = np.zeros(Am)
##dxs = np.linspace(0.1,0.8,Am)
##for it in range(Am):
##	print("Iteration ",it)
##	dx = dxs[it]
##	grid = bempp.api.shapes.sphere(h=dx)
##	RT_space = bempp.api.function_space(grid,"RT",0)
##	#def incident_field(x,n,domain_index,result):
##	#        #return np.array([np.exp(-50*(x[2]-t+2)**2), 0. * x[2], 0. * x[2]])
##	#        result[:] =  np.array([np.exp(-1*(x[2])), 0. * x[2], 0. * x[2]])
##	def a(x):
##		return np.linalg.norm(x)**(-0.5)*x
##
##	def tangential_trace(x, n, domain_index, result):
##	        result[:] = np.cross(n,np.cross(np.array([np.exp(-1*(x[2])), 0. * x[2], 0. * x[2]]), n))
##	def squared_tangtrace(x, n, domain_index, result):
##		trace     = np.cross(n,np.cross(np.array([np.exp(-1*(x[2])), 0. * x[2], 0. * x[2]]), n))
##	        result[:] = a(trace)
##	#print(np.cross(n,np.cross(np.array([np.exp(-1*(x[2])), 0. * x[2], 0. * x[2]]), n)))
##	#@bempp.api.real_callable
##	#def curl_trace(x,n,domain_index,result):
##	#        curlU=np.array([ 0. * x[2],-100*(x[2]-t+2)*np.exp(-50*(x[2]-t+2)**2), 0. * x[2]])
##	#        result[:] = np.cross(curlU , n)
##	trace_fun= bempp.api.GridFunction(RT_space, fun=tangential_trace,dual_space=RT_space)
##	trace_fun2= bempp.api.GridFunction(RT_space, fun=squared_tangtrace,dual_space=RT_space)
##	coefficientsSearched = trace_fun2.coefficients
##	coefficientsGiven = trace_fun.coefficients
##	dof = RT_space.global_dof_count
##	CoeffToEval = getCoefftoEval(RT_space)
##	trace_fun_given = bempp.api.GridFunction(RT_space, coefficients = coefficientsGiven, dual_space = RT_space)
##	centerEvalsGiven = trace_fun_given.evaluate_on_element_centers()
##	verticeEvalsGiven = trace_fun_given.evaluate_on_vertices()
##	for j in range(len(centerEvalsGiven[0,:])):
##		centerEvalsGiven[:,j] = a(centerEvalsGiven[:,j])
##	for j in range(len(verticeEvalsGiven[0,:])):
##		verticeEvalsGiven[:,j] = a(verticeEvalsGiven[:,j])
##	evalDataFlat = np.concatenate((np.ravel(centerEvalsGiven),np.ravel(verticeEvalsGiven)),axis = 0)
##	#print(np.linalg.norm(CoeffToEval.dot(coefficientsGiven)-evalDataFlat))
##	#coeffa = np.linalg.solve(CoeffToEvalSquare,[evalDataFlat[i] for i in indizes])
##	coeffa,res,rank,s = np.linalg.lstsq(CoeffToEval,evalDataFlat)
##	lstsqfun = bempp.api.GridFunction(RT_space, coefficients = coeffa, dual_space = RT_space)
##	print(coeffa.shape)
##	print(coefficientsSearched.shape)
##	errs[it] = np.max(np.abs(coeffa-coefficientsSearched))
##	l2errs[it] = (lstsqfun-trace_fun2).l2_norm()
##import matplotlib.pyplot as plt
##plt.loglog(dxs,dxs,linestyle='dashed')
##plt.loglog(dxs,errs)
##plt.loglog(dxs,l2errs)
##plt.show()
###print((np.linalg.norm(coefficientsSearched-coeffa))/np.linalg.norm(coefficientsSearched))
###print(np.linalg.norm(CoeffToEval.dot(coeffa)-evalDataFlat))
##
##coefficients = np.ones(dof)
##evalcentersMat = (CoeffToEval.dot(coefficients))
##
###print(CoeffToEval.shape)
###print(np.linalg.matrix_rank(CoeffToEval))
###print(len(centerdata[0,:]))
##evalcentersMat = np.reshape(evalcentersMat,(3,len(evalcentersMat)/3))
##trace_fun= bempp.api.GridFunction(RT_space, coefficients = coefficients)
##evalcenters = trace_fun.evaluate_on_element_centers()
##
###plt.spy(CoeffToEval)
###plt.show()
###trace_fun.plot()
###print(RT_space.global_dof_count)
###print(trace_fun.evaluate_on_vertices())
###print(len(coefficients))
###print(trace_fun.evaluate_on_element_centers())
###import inspect
###print(inspect.getsource(bempp.api.GridFunction))
###def squared_trace(x,n,domain_index,result):
###	result[:] = trace_fun.evaluate(x,n)
###print(inspect.getsource(bempp.api.GridFunction))
###squared_fun= bempp.api.GridFunction(RT_space, fun=squared_trace,dual_space=RT_space)
###print(dir(trace_fun))
##
###from bempp.core.assembly.function_projector import calculate_projection
###print("TYP : ",type(calculate_projection))
###print(calculate_projection.__doc__)
###trace_evals = trace_fun.evaluate_on_vertices()
###for j in range(len(trace_evals[0,:])):
###	trace_evals[:,j] = a(trace_evals[:,j])
###print("##########################################################################")
##coeffs  = trace_fun.coefficients
##coeffs2 = trace_fun2.coefficients
##
##trace_func_coeff = bempp.api.GridFunction(RT_space, coefficients=coeffs)
##
###trace_fun2.plot()
###curl_fun = bempp.api.GridFunction(RT_space, fun=curl_trace,dual_space=RT_space)
##def harmonic_calderon(s,b,grid):
##        points=np.array([[0],[0],[2]])
##        #normb=np.linalg.norm(b[0])+np.linalg.norm(b[1])+np.linalg.norm(b[2])
##        normb=np.max(np.abs(b))
##        bound=np.abs(s)**4*np.exp(-s.real)*normb
##        print("s: ",s, " maxb: ", normb, " bound : ", bound)
##        if bound <10**(-9):
##                print("JUMPED")
##                return np.zeros(3)
##        OrderQF = 8
##        
##        #tol= np.finfo(float).eps
##        bempp.api.global_parameters.quadrature.near.max_rel_dist = 2
##        bempp.api.global_parameters.quadrature.near.single_order =OrderQF-1
##        bempp.api.global_parameters.quadrature.near.double_order = OrderQF-1
##        
##        bempp.api.global_parameters.quadrature.medium.max_rel_dist =4
##        bempp.api.global_parameters.quadrature.medium.single_order =OrderQF-2
##        bempp.api.global_parameters.quadrature.medium.double_order =OrderQF-2
##
##
##        bempp.api.global_parameters.quadrature.far.single_order =OrderQF-3
##        bempp.api.global_parameters.quadrature.far.double_order =OrderQF-3
##
##        bempp.api.global_parameters.quadrature.double_singular = OrderQF
##        bempp.api.global_parameters.hmat.eps=10**-4
##
##        bempp.api.global_parameters.hmat.admissibility='strong'
#####    Define Spaces
##        NC_space=bempp.api.function_space(grid, "NC",0)
##        RT_space=bempp.api.function_space(grid, "RT",0)
##                
##        elec = -bempp.api.operators.boundary.maxwell.electric_field(RT_space, RT_space, NC_space,1j*s)
##        magn = -bempp.api.operators.boundary.maxwell.magnetic_field(RT_space, RT_space, NC_space, 1j*s)
##
##        identity2=bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, RT_space)
##
##        identity= -bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, NC_space)
##        dof=NC_space.global_dof_count
##        
##        trace_fun= bempp.api.GridFunction(RT_space, coefficients=b[0:dof],dual_space=RT_space)
########## End condition, by theoretical bound:
##        normb=trace_fun.l2_norm()
##        bound=np.abs(s)**3*np.exp(-s.real)*normb
##        if bound <10**(-8):
##                print("JUMPED")
##                return np.zeros(3)
##
##        zero_fun= bempp.api.GridFunction(RT_space,coefficients = b[dof:],dual_space=RT_space)
##        
##        #rhs=[trace_fun,zero_fun]
##        id_discrete=identity2.weak_form()
##        b[0:dof]=id_discrete*b[0:dof]
##
##        blocks=np.array([[None,None], [None,None]])
##
##        #blocks[0,0] = -elec.weak_form()+10*s**0.5*identity2.weak_form()
##        blocks[0,0] = -elec.weak_form()+0.1*s**0.5*identity2.weak_form()
##        blocks[0,1] =  magn.weak_form()-1.0/2*identity.weak_form()
##        blocks[1,0] = -magn.weak_form()-1.0/2*identity.weak_form()
##        blocks[1,1] = -elec.weak_form()
##
##
##        blocks=bempp.api.BlockedDiscreteOperator(blocks)
###       A_mat=bempp.api.as_matrix(blocks)
###       print("A_mat : ",A_mat)
###       e,D=np.linalg.eig(A_mat)
###       print("Eigs : ", e)
###       print("Cond : ", np.linalg.cond(A_mat))
####
####      trace_fun= bempp.api.GridFunction(multitrace.range_spaces[0], coefficients=b[0:dof],dual_space=multitrace.dual_to_range_spaces[0])
####
####      zero_fun= bempp.api.GridFunction(multitrace.range_spaces[1],coefficients = b[dof:],dual_space=multitrace.dual_to_range_spaces[1])
####
####      rhs=[trace_fun,zero_fun]
####
####      #print("Still living")
####      
##        #from bempp.api.linalg import gmres 
##        from scipy.sparse.linalg import gmres
##        print("Start GMRES : ")
###       def print_res(rk):
###               print("Norm of residual: "+ str(np.linalg.norm(rk)))
##        #print(np.linalg.norm(lambda_data))
##        #lambda_data,info = gmres(blocks, b,tol=10**-4,restart=50,maxiter=100,callback=print_res)
##        lambda_data,info = gmres(blocks, b,tol=10**-5,maxiter=300)
##        print("INFO :", info)
##        #lambda_data,info = gmres(blocks, b,tol=10**-4,callback=print_res)
##        #print("I survived!")
##        #from bempp.api.linalg import lu
##        #lambda_data = lu(elec, trace_fun)
##        #lambda_data.plot()
##        #print("Norm lambda_data : ",np.linalg.norm(lambda_data))
##        #if (np.linalg.norm(lambda_data)<10**-10):
##        phigrid=bempp.api.GridFunction(RT_space,coefficients=lambda_data[0:dof],dual_space=RT_space)
##        psigrid=bempp.api.GridFunction(RT_space,coefficients=lambda_data[dof:2*dof],dual_space=RT_space)
##
##        slp_pot = bempp.api.operators.potential.maxwell.electric_field(RT_space, points, s*1j)
##
##        dlp_pot = bempp.api.operators.potential.maxwell.magnetic_field(RT_space, points, s*1j)
##        print("Evaluate field : ")      
##        scattered_field_data = -slp_pot * phigrid+dlp_pot*psigrid
###       scattered_field_H = -slp_pot * psigrid-dlp_pot*phigrid
###       H = scattered_field_H.reshape(3,1)[:,0]
###       print(" E : ", E, " H : ", H)
###       print("Angle: ", np.dot(E,H), " Scalar product with conjugation : ", np.dot(np.conj(E),H))
###       print("NORM COMBINED OPERATOR :" , np.linalg.norm(scattered_field_data)/np.linalg.norm(b))
###       print(scattered_field_data)
###       print("NORM ScatteredField :", np.linalg.norm(scattered_field_data))
###
###       print("s : ", s)
###       print("NORM B :" ,np.linalg.norm(b))
##        if np.isnan(scattered_field_data).any():
##                print("NAN Warning, s = ", s)
##                scattered_field_data=np.zeros(np.shape(scattered_field_data))
##        return scattered_field_data.reshape(3,1)[:,0]
