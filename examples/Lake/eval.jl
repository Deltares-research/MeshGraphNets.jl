cd(@__DIR__)
using Pkg
Pkg.activate("../..")

using MeshGraphNets

import OrdinaryDiffEq: Euler, Tsit5
import Optimisers: Adam

######################
# Network parameters #
######################

message_steps = 15
layer_size = 128
hidden_layers = 2
batch = 1
epo = 1
ns = 5
norm_steps = 0
cuda = true
cp_derivative = 20
cp_solver = 10
ad = :Zygote

########################
# Node type parameters #
########################

types_inflow = [4]
types_updated = [0, 5]
types_noisy = [0]
noise_stddevs = [0.02f0]

learning_rate_derivative = 1.0f-4
opt_derivative = Adam(learning_rate_derivative)

ds_path = "./data"
chk_path = "./chk"
eval_path = "./eval"

tstart = 0.0f0
dt = 10000
tstop = 180000

# timesteps at which the mean squared error is calculated and printed during evaluation
mse_steps = vcat(collect(tstart:dt:tstop), tstop)

eval_network(
    ds_path, chk_path, eval_path, Euler(); start = tstart, stop = tstop,
    dt = dt, saves = tstart:dt:tstop, mse_steps = collect(mse_steps), mps = message_steps,
    layer_size = layer_size, hidden_layers = hidden_layers, use_cuda = cuda
)