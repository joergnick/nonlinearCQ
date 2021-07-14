import scipy.io
import bempp.api
import numpy as np
from scipy.sparse.linalg import aslinearoperator
#grid = bempp.api.shapes.sphere(h=1)
#

mat_contents=scipy.io.loadmat('grids/TorusDOF294.mat')
#mat_contents=scipy.io.loadmat('grids/TorusDOF896.mat')
Nodes=np.array(mat_contents['Nodes']).T
rawElements=mat_contents['Elements']
for j in range(len(rawElements)):
	betw=rawElements[j][0]
	rawElements[j][0]=rawElements[j][1]
	rawElements[j][1]=betw
Elements=np.array(rawElements).T
## Shifting smallest Index to 0.
Elements=Elements-1
AmElements = len(Elements[0,:])

#print(AmElements)
grid=bempp.api.grid_from_element_data(Nodes,Elements)

RT_space = bempp.api.function_space(grid,"RT",0)
dof = RT_space.global_dof_count
coeffs = np.zeros(dof)
coeffs[0] = 1
one_grid = bempp.api.GridFunction(RT_space,coefficients = coeffs)
two_grid = bempp.api.GridFunction(RT_space,coefficients = coeffs)

## START INTEGRATION
from bempp.api.integration import gauss_triangle_points_and_weights
import numpy as np

components = one_grid.component_count
res = np.zeros((components, 1), dtype='float64')
accuracy_order = one_grid.parameters.quadrature.far.single_order
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
    print("HI,points: ",points, " evals: ",one_grid.evaluate(element, points))
    res += np.sum(
        np.dot(one_grid.evaluate(element, points).T,two_grid.evaluate(element, points))*weights * integration_elements,
        axis=1)


print(res)
## END INTEGRATION
print(one_grid.integrate())
center_evals = one_grid.evaluate_on_element_centers()
#print(len(center_evals[0,:]))
identity = bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, RT_space)
id_weak = identity.weak_form()
id_sparse = aslinearoperator(id_weak).sparse_operator
id_dense = id_sparse.A
print(id_dense[0,0])
import matplotlib.pyplot as plt

#print(id_dense)
