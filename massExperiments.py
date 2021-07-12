import bempp.api
from scipy.sparse.linalg import aslinearoperator
grid = bempp.api.shapes.sphere(h=1)
RT_space = bempp.api.function_space(grid,"RT",0)
identity = bempp.api.operators.boundary.sparse.identity(RT_space, RT_space, RT_space)
id_weak = identity.weak_form()
import inspect
id_sparse = aslinearoperator(id_weak).sparse_operator

id_dense = id_sparse.A
print(id_dense)
