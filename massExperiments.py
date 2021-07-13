import bempp.api
from scipy.sparse.linalg import aslinearoperator
#grid = bempp.api.shapes.sphere(h=1)
#
mat_contents=scipy.io.loadmat('grids/TorusDOF896.mat')
Nodes=np.array(mat_contents['Nodes']).T
rawElements=mat_contents['Elements']
for j in range(len(rawElements)):
	betw=rawElements[j][0]
	rawElements[j][0]=rawElements[j][1]
	rawElements[j][1]=betw
Elements=np.array(rawElements).T
Elements=Elements-1
grid=bempp.api.grid_from_element_data(Nodes,Elements)

RT_space = bempp.api.function_space(grid,"RT",0)
identity = bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, RT_space)
id_weak = identity.weak_form()
import inspect
id_sparse = aslinearoperator(id_weak).sparse_operator
id_dense = id_sparse.A

print(id_dense)
