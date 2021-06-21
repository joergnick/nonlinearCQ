# nonlinearCQ
Project implements a black box CQ solver for nonlinear problems, targeted at scattering problems by making use of the BEM - Library Bempp. The toolbox is capable of solving a general class of evolution equations, assuming vanishing initial conditions (for both the value and its derivatives). The abstract form of the evolution equation discretized is given by

<img src="https://render.githubusercontent.com/render/math?math=B(\partial_t)u-a(u)=f(t)">.

In particular this gives a discretization of fractional evolution equations.
