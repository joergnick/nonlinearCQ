import scipy.io
import bempp.api
import numpy as np
from scipy.sparse.linalg import aslinearoperator
grid = bempp.api.shapes.sphere(h=0.5)
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
    identity = bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, RT_space)
    id_weak = identity.weak_form()
    from scipy.sparse.linalg import gmres
    coeffsol,info = gmres(id_weak,weightIntegrals,tol=10**(-5))
    return bempp.api.GridFunction(gridFun.space,coefficients = coeffsol)


RT_space = bempp.api.function_space(grid,"RT",0)
gridfunList,neighborlist,domainDict    = precompMM(RT_space)
def a(x):
    return np.linalg.norm(x)**2*x
def u(x, n, domain_index, result):
        result[:] = np.cross(n,np.cross(np.array([np.exp(-1*(x[2])), 0. * x[2], 0. * x[2]]), n))

u_gridFun  = bempp.api.GridFunction(RT_space, fun=tangential_trace,dual_space=RT_space)
a_of_u_gridFun = applyNonlinearity(trace_fun,a,gridfunList,domainDict)

def atang_trace(x,n,domain_index,result):
    result[:] = a(np.cross(n,np.cross(np.array([np.exp(-1*(x[2])), 0. * x[2], 0. * x[2]]), n)))

exatrace_fun  = bempp.api.GridFunction(RT_space, fun=atang_trace,dual_space=RT_space)

exatrace_fun.plot()






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


