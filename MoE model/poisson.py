#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from new_model import MoEPINN, vdp_residual

torch.manual_seed(0)

#%%
a = 3
mu = 0.3
c = 10

f_sharp = lambda x: a * torch.exp(-(x-mu)**2/2*c**2)
f_smooth = lambda x: torch.sin(2*x)

f = lambda x: f_sharp(x) + f_smooth(x)

x = torch.linspace(0,1,100)

plt.plot(x, f(x))

#%%
t0 = 0
T = 1


# %%
# Numerical reference solution
