import numpy as np
import bempp.api
import inspect
import matplotlib.pyplot as plt
dx = 1
grid = bempp.api.shapes.sphere(h=dx)
RT_space = bempp.api.function_space(grid,"RT",0)
#def incident_field(x,n,domain_index,result):
#        #return np.array([np.exp(-50*(x[2]-t+2)**2), 0. * x[2], 0. * x[2]])
#        result[:] =  np.array([np.exp(-1*(x[2])), 0. * x[2], 0. * x[2]])
def a(x):
	return np.linalg.norm(x)**(-0.5)*x
#print(np.cross(n,np.cross(np.array([np.exp(-1*(x[2])), 0. * x[2], 0. * x[2]]), n)))
#@bempp.api.real_callable
#def curl_trace(x,n,domain_index,result):
#        curlU=np.array([ 0. * x[2],-100*(x[2]-t+2)*np.exp(-50*(x[2]-t+2)**2), 0. * x[2]])
#        result[:] = np.cross(curlU , n)
#print(inspect.getsource(bempp.api.GridFunction.from_random))
dof = RT_space.global_dof_count
samples = dof
cMat = np.zeros((dof,dof))
for j in range(samples):
	cMat[j,:] = bempp.api.GridFunction.from_random(RT_space).coefficients
E = np.linalg.eigvals(cMat)
print(np.sort(np.abs(E)))
