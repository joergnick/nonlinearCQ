import unittest
import numpy as np
from cqRK import CQModel
class ScatModel(CQModel):
	def precomputing(self,s):
		return s**2
	def harmonicForward(self,s,b,precomp = None):
		return precomp*b
	def righthandside(self,t,history = None):
		return 8*7*t**6
		#return 3*t**2+t**9
	def nonlinearity(self,x):
		return 0*x
		#return x**3

class TestCQMethods(unittest.TestCase):
	def setUp(self):
		self.model = ScatModel()
		self.T = 1
		self.N = 8
		self.dof = 1
		self.ex_sol = np.linspace(0,self.T,self.N+1)**8
	def test_LinearRadauIIA_2(self):
		m = 2
		method = "RadauIIA-"+str(m)
		sol = self.model.simulate(self.T,self.N,self.dof,method = method)
		err = max(np.abs(sol[0,::m]-self.ex_sol))
		self.assertLess(np.abs(err),10**(-2))
	def test_LinearRadauIIA_3(self):
		m = 3
		method = "RadauIIA-"+str(m)
		sol = self.model.simulate(self.T,self.N,self.dof,method = method)
		err = max(np.abs(sol[0,::m]-self.ex_sol))
		self.assertLess(np.abs(err),10**(-4))

if __name__ == '__main__':
	unittest.main()

