import numpy as np

dof = 2
m = 2
Delta0 = np.array([[3, 1],[-9, 5]])
deltaeigs,T = np.linalg.eig(Delta0)
Tinv = np.linalg.inv(T)
rhs_fft = np.array([[ 1+1j,2+100*1j],[4,5]])
rhsStages = 1j*np.zeros((dof,m))
for stageInd in range(m):
	for sumInd in range(m):
		rhsStages[:,stageInd]=rhsStages[:,stageInd]+Tinv[stageInd,sumInd]*rhs_fft[:,sumInd]


print(rhsStages)

rhsStages = 1j*np.zeros((dof,m))
rhsStages = np.matmul(rhs_fft,Tinv.T)
print(rhsStages)
