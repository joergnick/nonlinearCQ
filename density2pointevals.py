import numpy as np
import bempp.api
#grid = bempp.api.shapes.sphere(h=2**(0))
grid = bempp.api.shapes.sphere(h=2**(-0/2))
RT_space = bempp.api.function_space(grid,"RT",0)
dof = RT_space.global_dof_count
#sol = np.load('data/solh1.0N64m2.npy')
sol = np.load('data/solh1.0N30m2.npy')
m=2
print("Does sol contain Nan values? Answer: "+str(np.isnan(sol).any()))
N = (len(sol[0,:])-1)/2
print("N= ",N)
dof = len(sol[:,0])/2
print("DOF = ",dof)
#gridphi = bempp.api.GridFunction(RT_space,coefficients = sol[:dof,100])
#gridphi.plot()
#import matplotlib.pyplot as plt
#plt.imshow(sol[:dof,:])
#plt.colorbar()
#plt.show()
#### Create Points
#x_a=-0.75
#x_b=0.75
#y_a=-0.25
#y_b=1.25

x_a=-2
x_b=2
y_a=-2
y_b=2
n_grid_points= 150
nx = n_grid_points
nz = n_grid_points
############################################
plot_grid = np.mgrid[y_a:y_b:1j*n_grid_points, x_a:x_b:1j*n_grid_points]
#plot_grid = np.mgrid[-0.5:1:1j*n_grid_points, -1.5:1.5:1j*n_grid_points]
#print(plot_grid)
#points = np.vstack( ( plot_grid[0].ravel() , plot_grid[1].ravel() , 0.25*np.ones(plot_grid[0].size) ) )
points = np.vstack( ( plot_grid[0].ravel()  , 0*np.ones(plot_grid[0].size) , plot_grid[1].ravel()) )
radius = points[0,:]**2+points[1,:]**2+points[2,:]**2
def kirchhoff_repr(s,lambda_data):
    print("norm(density)=",np.linalg.norm(lambda_data))
    if np.linalg.norm(lambda_data)<10**(-3):
        print("Jumped")
        return np.zeros(n_grid_points**2*3)
    phigrid=bempp.api.GridFunction(RT_space,coefficients=lambda_data[0:dof],dual_space=RT_space)
    psigrid=bempp.api.GridFunction(RT_space,coefficients=lambda_data[dof:2*dof],dual_space=RT_space)
    slp_pot = bempp.api.operators.potential.maxwell.electric_field(RT_space, points, s*1j)
    dlp_pot = bempp.api.operators.potential.maxwell.magnetic_field(RT_space, points, s*1j)
    scattered_field_data = -slp_pot * phigrid+dlp_pot*psigrid
    scattered_field_data[np.isnan(scattered_field_data)] = 0 
    return scattered_field_data.reshape(n_grid_points**2*3,1)[:,0]
from linearcq import Conv_Operator
mSpt_Dpt = Conv_Operator(kirchhoff_repr)
T=4
uscatStages = mSpt_Dpt.apply_RKconvol(sol,T,method = "RadauIIA-2")
uscat = uscatStages[:,::2]
#uscat = np.zeros((n_grid_points**2*3,N))
uscat = np.concatenate((np.zeros((len(uscat[:,0]),1)),uscat),axis = 1)
import matplotlib
from matplotlib import pylab as plt 
u_ges=np.zeros((n_grid_points**2,N+1))
for j in range(N+1):    
    # Adjust the figure size in IPython
    matplotlib.rcParams['figure.figsize'] = (10.0, 8.0) 
    t=j*T*1.0/N
    def incident_field(x):
        return np.array([np.exp(-50*(x[2]-t+2)**2), 0. * x[2], 0. * x[2]])
    incident_field_data = incident_field(points)
    #scat_eval=np.zeros(nx*nz*3)
    #incident_field_data[radius<1]=np.nan
    scat_eval=uscat[:,j].reshape(3,nx*nz)
    #print(scat_eval)
    field_data = scat_eval + incident_field_data
    #field_data = scat_eval 
    #field_data = incident_field_data 
    squared_field_density = np.real(np.sum(field_data * field_data,axis = 0))
    u_ges[:,j]=squared_field_density.T
    #squared_field_density=field_data[2,:]
    squared_field_density[radius<1]=np.nan
    #squared_field_density[radius<1]=np.nan
    plt.imshow(squared_field_density.reshape((nx, nz)).T,
               cmap='coolwarm', origin='lower',
               extent=[x_a, x_b, y_a, y_b])
    plt.clim(vmin=0,vmax=1.0)
    plt.title("Squared Electric Field Density")
    plt.savefig("data/wave_images/Screen_n{}.png".format(j))
    if j==10:
        plt.colorbar()
    plt.clim((-1,1))

import scipy.io
scipy.io.savemat('data/h01N30.mat',dict(u_ges=u_ges,N=N,T=T,plot_grid=plot_grid,points=points))
