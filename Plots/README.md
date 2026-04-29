Standard parameters for PINN (Van der Pol):
# Parameters
t0 = 0.0
T = 20.0
mu = 5.0

x0 = 2.0
v0 = 0.0

# Training setup
n_epoch = 10_000
N = 800 # Uniform sampling

# Initialize network and optimizer
net = PINN(in_dim=1, out_dim=1, hidden_dim=64, hidden_layers=2)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# Loss term
w_pde = 1.0
w_ic = 10.0
w_data = 3.0