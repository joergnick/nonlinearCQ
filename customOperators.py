import scipy.io
import bempp.api
import numpy as np
from scipy.sparse.linalg import aslinearoperator
#grid = bempp.api.shapes.sphere(h=0.5)
##mat_contents=scipy.io.loadmat('grids/TorusDOF294.mat')
##mat_contents=scipy.io.loadmat('grids/TorusDOF896.mat')
##Nodes=np.array(mat_contents['Nodes']).T
##rawElements=mat_contents['Elements']
##for j in range(len(rawElements)):
##    betw=rawElements[j][0]
##    rawElements[j][0]=rawElements[j][1]
##    rawElements[j][1]=betw
##Elements=np.array(rawElements).T
#### Shifting smallest Index to 0.
##Elements=Elements-1
##AmElements = len(Elements[0,:])
##
###print(AmElements)
##grid=bempp.api.grid_from_element_data(Nodes,Elements)
##print(dict(grid))
#
#RT_space = bempp.api.function_space(grid,"RT",0)
def precompMM(space):
    nontrivialEntries = []
    dof = space.global_dof_count
    import numpy as np
    coeffs = np.zeros(dof)
    element_list =  list(space.grid.leaf_view.entity_iterator(0))
    gridfunList = []
    domainList = [[] for _ in element_list]
    domainDict = dict(zip(element_list,[[] for _ in element_list]))
    for j in range(dof):
        coeffs[j] = 1
        gridfun = bempp.api.GridFunction(space,coefficients = coeffs)
        gridfunList.append(gridfun)
        indices = np.nonzero(np.sum(gridfun.evaluate_on_element_centers()**2,axis = 0))
        domainList[indices[0][0]].append(j)
        domainList[indices[0][1]].append(j)
        domainDict[element_list[indices[0][0]]].append(j)
        domainDict[element_list[indices[0][1]]].append(j)
        coeffs[j] = 0
    identity = bempp.api.operators.boundary.sparse.identity(space,space,space)
    id_weak = identity.weak_form()
    id_sparse = aslinearoperator(id_weak).sparse_operator
    neighborlist = id_sparse.tolil().rows
    return gridfunList,neighborlist,domainDict

def sparseWeightedMM(space,weightGF,Da,gridfunList,neighborlist,domainDict):
    import time
    #print(weightGF)
    dof = space.global_dof_count
    import numpy as np
    from bempp.api.integration import gauss_triangle_points_and_weights
    accuracy_order = gridfunList[0].parameters.quadrature.far.single_order
    points, weights = gauss_triangle_points_and_weights(accuracy_order)
    element_list = list(gridfunList[0].grid.leaf_view.entity_iterator(0))
    data = []
    row  = []
    col  = []
    massdense = np.zeros((dof,dof))
    for element in element_list:
        integration_elements = element.geometry.integration_elements(points)
        for i in domainDict[element]:
            for j in domainDict[element]:
                data.append(0)
                row.append(i)
                col.append(j)
                A = gridfunList[i].evaluate(element, points)
                B = gridfunList[j].evaluate(element, points)
                #print(weightGF)
                #print(type(weightGF))
                weightGFeval = weightGF.evaluate(element,points)
                data[-1] += sum([A[:,k].dot(Da(weightGFeval[:,k]).dot(B[:,k]))*weights[k]*integration_elements[k] for k in range(len(A[0,:]))] ) 
    return scipy.sparse.csc_matrix((data,(row,col)))

def applyNonlinearity(gridFun,nonlinearity,gridfunList,domainDict):
    space = gridFun.space
    dof = space.global_dof_count
    coeff = gridFun.coefficients
    from bempp.api.integration import gauss_triangle_points_and_weights
    accuracy_order = gridFun.parameters.quadrature.far.single_order
    points, weights = gauss_triangle_points_and_weights(accuracy_order)
    element_list = list(gridfunList[0].grid.leaf_view.entity_iterator(0))
    weightIntegrals = np.zeros(dof)
    for element in element_list:
        integration_elements = element.geometry.integration_elements(points)
        for i in domainDict[element]:
           testFuncEvals = gridfunList[i].evaluate(element, points)
           gFevals = gridFun.evaluate(element,points)
           weightIntegrals[i] += sum([testFuncEvals[:,k].dot(nonlinearity(gFevals[:,k]))*weights[k]*integration_elements[k] for k in range(len(testFuncEvals[0,:]))] )
    identity = bempp.api.operators.boundary.sparse.identity(space, space, space)
    id_weak = identity.weak_form()
    from scipy.sparse.linalg import gmres
    coeffsol,info = gmres(id_weak,weightIntegrals,tol=10**(-5))
    return bempp.api.GridFunction(gridFun.space,coefficients = coeffsol)


def sparseMM(space,gridfunList,neighborlist,domainDict):
    import time
    dof = space.global_dof_count
    import numpy as np
    from bempp.api.integration import gauss_triangle_points_and_weights
    accuracy_order = gridfunList[0].parameters.quadrature.far.single_order
    points, weights = gauss_triangle_points_and_weights(accuracy_order)
    element_list = list(gridfunList[0].grid.leaf_view.entity_iterator(0))
    data = []
    row  = []
    col  = []
    massdense = np.zeros((dof,dof))
    for element in element_list:
        integration_elements = element.geometry.integration_elements(points)
        for i in domainDict[element]:
            for j in domainDict[element]:
                data.append(0)
                row.append(i)
                col.append(j)
                A = gridfunList[i].evaluate(element, points)
                B = gridfunList[j].evaluate(element, points)
                data[-1] += sum([A[:,k].dot(B[:,k])*weights[k]*integration_elements[k] for k in range(len(A[0,:]))] ) 
    return scipy.sparse.csc_matrix((data,(row,col)))



def massMatrix(space,gridfunList,neighborlist,domainDict):
    import time
    dof = space.global_dof_count
    import numpy as np
    from bempp.api.integration import gauss_triangle_points_and_weights
    accuracy_order = gridfunList[0].parameters.quadrature.far.single_order
    points, weights = gauss_triangle_points_and_weights(accuracy_order)
    element_list = list(gridfunList[0].grid.leaf_view.entity_iterator(0))
    massdense = np.zeros((dof,dof))
    for element in element_list:
        integration_elements = element.geometry.integration_elements(points)
        for i in domainDict[element]:
            for j in domainDict[element]:
                A = gridfunList[i].evaluate(element, points)
                B = gridfunList[j].evaluate(element, points)
                massdense[i,j] += sum([A[:,k].dot(B[:,k])*weights[k]*integration_elements[k] for k in range(len(A[0,:]))] ) 
    return massdense

#import time
#start = time.time()
#gridfunList,neighborlist,domainDict    = precompMM(RT_space)
#precomptime = time.time()
#def a(x):
#    return np.linalg.norm(x)**2*x
#    #return np.linalg.norm(x)**(-0.5)*x
#def tangential_trace(x, n, domain_index, result):
#        result[:] = np.cross(n,np.cross(np.array([np.exp(-1*(x[2])), 0. * x[2], 0. * x[2]]), n))
#def atang_trace(x,n,domain_index,result):
#    result[:] = a(np.cross(n,np.cross(np.array([np.exp(-1*(x[2])), 0. * x[2], 0. * x[2]]), n)))
##
#trace_fun  = bempp.api.GridFunction(RT_space, fun=tangential_trace,dual_space=RT_space)
#exatrace_fun  = bempp.api.GridFunction(RT_space, fun=atang_trace,dual_space=RT_space)
#
#approxatrace_fun = applyNonlinearity(trace_fun,a,gridfunList,domainDict)
##exatrace_fun.plot()
##(approxatrace_fun-exatrace_fun).plot()
#print("DIFFERENCE IN GRIDFUNCTIONS: ",(exatrace_fun-approxatrace_fun).l2_norm())
##print("Precomputing time: ",precomptime-start)
#M = massMatrix(RT_space,gridfunList,neighborlist,domainDict)
#Msparse = sparseMM(RT_space,gridfunList,neighborlist,domainDict)
#def Da(x):
#    return np.eye(3)
#testGridFun = bempp.api.GridFunction.from_ones(RT_space)
#Msparse = sparseWeightedMM(RT_space,testGridFun,Da,gridfunList,neighborlist,domainDict)
#print(type(Msparse))
#assemblytime = time.time()
##print("Assembly time: ",assemblytime-precomptime)
#identity = bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, RT_space)
##import inspect
##print(inspect.getsource(identity.__class__.__base__))
##print(inspect.getsource(type(identity)))
#id_weak = identity.weak_form()
#id_sparse = aslinearoperator(id_weak).sparse_operator
#id_dense = id_sparse.A
##nprint("Bempp assembly time: ", time.time()-assemblytime)
#print("Difference in Matrix: ",np.linalg.norm(id_dense-Msparse.A))
#
##print(M)
#
##dof = RT_space.global_dof_count
##coeffs = np.zeros(dof)
##coeffs[0] = 1
##one_grid = bempp.api.GridFunction(RT_space,coefficients = coeffs)
##coeffs[0] = 0
##coeffs[1] = 1
##two_grid = bempp.api.GridFunction(RT_space,coefficients = coeffs)
##
#### START INTEGRATION
##from bempp.api.integration import gauss_triangle_points_and_weights
##import numpy as np
##
##components = one_grid.component_count
###res = np.zeros((components, 1), dtype='float64')
##res = 0
##accuracy_order = one_grid.parameters.quadrature.far.single_order
##accuracy_order = 5
##points, weights = gauss_triangle_points_and_weights(accuracy_order)
#### Additional line:
##element = None
#### Resume integrate method
##element_list = [element] if element is not None else list(
##    one_grid.grid.leaf_view.entity_iterator(0))
###print(dir(element_list[0].geometry.local2global([0,0,1])))
###print(element_list[0].geometry.local2global(points))
###print(inspect.getsource(element_list[0].geometry.local2global))
##
##for element in element_list:
##    integration_elements = element.geometry.integration_elements(
##        points)
##    A = one_grid.evaluate(element, points)
##    B = two_grid.evaluate(element, points)
##    res += sum([A[:,i].dot(B[:,i])*weights[i]*integration_elements[i] for i in range(len(A[0,:]))] ) 
##    #res += np.sum(np.matmul(one_grid.evaluate(element, points).T,two_grid.evaluate(element, points))*weights * integration_elements,
##    #    axis=1)
##
#### END INTEGRATION
###print(one_grid.integrate())
##center_evals = one_grid.evaluate_on_element_centers()
###print(len(center_evals[0,:]))
##identity = bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, RT_space)
#import matplotlib.pyplot as plt
#
#
