import scipy.io
import bempp.api
import numpy as np
from scipy.sparse.linalg import aslinearoperator
grid = bempp.api.shapes.sphere(h=0.5)
#mat_contents=scipy.io.loadmat('grids/TorusDOF294.mat')
#mat_contents=scipy.io.loadmat('grids/TorusDOF896.mat')
#Nodes=np.array(mat_contents['Nodes']).T
#rawElements=mat_contents['Elements']
#for j in range(len(rawElements)):
#    betw=rawElements[j][0]
#    rawElements[j][0]=rawElements[j][1]
#    rawElements[j][1]=betw
#Elements=np.array(rawElements).T
### Shifting smallest Index to 0.
#Elements=Elements-1
#AmElements = len(Elements[0,:])
#
##print(AmElements)
#grid=bempp.api.grid_from_element_data(Nodes,Elements)
#print(dict(grid))


RT_space = bempp.api.function_space(grid,"RT",0)
def precompMM(space):
    nontrivialEntries = []
    dof = space.global_dof_count
    #for j in range
    import numpy as np
    import bempp.api 
    identity = bempp.api.operators.boundary.sparse.identity(space,space,space)
    id_weak = identity.weak_form()
    id_sparse = aslinearoperator(id_weak).sparse_operator
    return id_sparse
#print(dir(precompMM(RT_space)))
#print(precompMM(RT_space))
#print(precompMM(RT_space).indices)
#print(precompMM(RT_space).indptr)
#print(len(precompMM(RT_space).indices))
#print(len(precompMM(RT_space).indptr))
space = RT_space
identity = bempp.api.operators.boundary.sparse.identity(space,space,space)
id_weak = identity.weak_form()
id_sparse = aslinearoperator(id_weak).sparse_operator
M = id_sparse.A
def massMatrix(space):
    dof = space.global_dof_count
    import numpy as np
    coeffs = np.zeros(dof)
    gridfunList = []
    for j in range(dof):
        coeffs[j] = 1
        gridfunList.append(bempp.api.GridFunction(space,coefficients = coeffs))
        coeffs[j] = 0
    from bempp.api.integration import gauss_triangle_points_and_weights
    accuracy_order = gridfunList[0].parameters.quadrature.far.single_order
    points, weights = gauss_triangle_points_and_weights(accuracy_order)
    element = None
    ## Resume integrate method
    element_list = [element] if element is not None else list(
    gridfunList[0].grid.leaf_view.entity_iterator(0))
    massdense = np.zeros((dof,dof))
    identity = bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, RT_space)
    id_weak = identity.weak_form()
    id_sparse = aslinearoperator(id_weak).sparse_operator
    id_dense = id_sparse.A   
    neighborlist = precompMM(RT_space).tolil().rows
    #print(len(gridfunList[0].evaluate_on_element_centers()[0,:]))
    for element in element_list:
        integration_elements = element.geometry.integration_elements(points)
        for i in range(dof):
            for j in neighborlist[i]:
                A = gridfunList[i].evaluate(element, points)
                B = gridfunList[j].evaluate(element, points)
                massdense[i,j] += sum([A[:,k].dot(B[:,k])*weights[k]*integration_elements[k] for k in range(len(A[0,:]))] ) 
    return massdense

print("GLOBAL DOF : ", RT_space.global_dof_count)
M = massMatrix(RT_space)
#print(M)

dof = RT_space.global_dof_count
coeffs = np.zeros(dof)
coeffs[0] = 1
one_grid = bempp.api.GridFunction(RT_space,coefficients = coeffs)
coeffs[0] = 0
coeffs[1] = 1
two_grid = bempp.api.GridFunction(RT_space,coefficients = coeffs)

## START INTEGRATION
from bempp.api.integration import gauss_triangle_points_and_weights
import numpy as np

components = one_grid.component_count
#res = np.zeros((components, 1), dtype='float64')
res = 0
accuracy_order = one_grid.parameters.quadrature.far.single_order
accuracy_order = 5
points, weights = gauss_triangle_points_and_weights(accuracy_order)
## Additional line:
element = None
## Resume integrate method
element_list = [element] if element is not None else list(
    one_grid.grid.leaf_view.entity_iterator(0))
#print(dir(element_list[0].geometry.local2global([0,0,1])))
#print(element_list[0].geometry.local2global(points))
#print(inspect.getsource(element_list[0].geometry.local2global))

for element in element_list:
    integration_elements = element.geometry.integration_elements(
        points)
    A = one_grid.evaluate(element, points)
    B = two_grid.evaluate(element, points)
    res += sum([A[:,i].dot(B[:,i])*weights[i]*integration_elements[i] for i in range(len(A[0,:]))] ) 
    #res += np.sum(np.matmul(one_grid.evaluate(element, points).T,two_grid.evaluate(element, points))*weights * integration_elements,
    #    axis=1)

## END INTEGRATION
#print(one_grid.integrate())
center_evals = one_grid.evaluate_on_element_centers()
#print(len(center_evals[0,:]))
identity = bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, RT_space)
id_weak = identity.weak_form()
id_sparse = aslinearoperator(id_weak).sparse_operator
id_dense = id_sparse.A
print(np.linalg.norm(id_dense-M))
import matplotlib.pyplot as plt


