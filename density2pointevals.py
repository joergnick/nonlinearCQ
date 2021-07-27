import numpy as np
import bempp.api
grid = bempp.api.shapes.sphere(h=1)
RT_space = bempp.api.function_space(grid,"RT",0)
dof = RT_space.global_dof_count
sol = np.load('data/sol.npy')
#### Create Points
#x_a=-0.75
#x_b=0.75
#y_a=-0.25
#y_b=1.25

x_a=-2
x_b=2
y_a=-2
y_b=2
n_grid_points=150
############################################
plot_grid = np.mgrid[y_a:y_b:1j*n_grid_points, x_a:x_b:1j*n_grid_points]
#plot_grid = np.mgrid[-0.5:1:1j*n_grid_points, -1.5:1.5:1j*n_grid_points]
#print(plot_grid)
#points = np.vstack( ( plot_grid[0].ravel() , plot_grid[1].ravel() , 0.25*np.ones(plot_grid[0].size) ) )
points = np.vstack( ( plot_grid[0].ravel()  , 0*np.ones(plot_grid[0].size) , plot_grid[1].ravel()) )

def kirchhoff_repr(s,b):
    phigrid=bempp.api.GridFunction(RT_space,coefficients=lambda_data[0:dof],dual_space=RT_space)
    psigrid=bempp.api.GridFunction(RT_space,coefficients=lambda_data[dof:2*dof],dual_space=RT_space)
    slp_pot = bempp.api.operators.potential.maxwell.electric_field(RT_space, points, s*1j)
    dlp_pot = bempp.api.operators.potential.maxwell.magnetic_field(RT_space, points, s*1j)
    scattered_field_data = -slp_pot * phigrid+dlp_pot*psigrid
    return scattered_field_data.reshape(n_grid_points**2*3,1)[:,0]
from linearcq import Conv_Operator
