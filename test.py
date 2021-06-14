from cq import CQModel
import numpy as np
model = CQModel()
T = 1
Am = 5
taus = np.zeros(Am)
err = np.zeros(Am)
for j in range(Am):
	N = 4*2**j
	taus[j] = T*1.0/N
	tt = np.linspace(0,T,N+1)
	ex_sol = 1.0/30*tt**6
	#ex_sol = 4*3*tt**2
	sol = model.simulate(T,N,1)
	err[j] = max(np.abs(sol[0,:]-ex_sol))

import matplotlib.pyplot as plt
plt.loglog(taus,err)
plt.loglog(taus,taus**2,linestyle = 'dashed')
plt.show()
	
