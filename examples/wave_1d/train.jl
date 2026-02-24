cd(@__DIR__)
using Pkg
Pkg.activate("../..")

using MeshGraphNets

import OrdinaryDiffEq: Euler, Tsit5
import Optimisers: Adam
import ParameterSchedulers: Step
import Lux: swish

######################
# Network parameters #
######################

message_steps = 7
layer_size = 128
hidden_layers = 2
activation_function = swish # activation function used in the mlp (see Lux docs for available ones)
epo = 1 # Number times full train set is looped over
ns = 30*360 # Total number of steps seen during training, overrides epochs if epochs is set too low
nbatch = 4
norm_steps = 0 # number of steps to wait at start of training before applying model gradient
train_noiseless = 360 # number of steps train loss is calced for without input noise. Also determines no. of train trajectories being rolled out
cuda = true
cp_derivative = ns÷4
cp_solver = 10
ad = :Zygote

########################
# Node type parameters #
########################

types_inflow = [4]
types_updated = [0, 5]
types_noisy = [0]
noise_stddevs = [0.02f0]

# Learning rate scheduling makes use of ParameterSchedulers.jl
# See docs for available schedulers and their syntax
learning_rate_derivative = 1.0f-4 # starting learning rate
lr_decay = 0.5f0 # learning rate update factor
lr_decay_rate = ns÷4 # after how many steps learning rate is updated
opt_derivative = Adam(learning_rate_derivative)
schedule = Step(start=learning_rate_derivative, decay=lr_decay, step_sizes=lr_decay_rate)

train_strat = DerivativeBatchTraining(batch_size=nbatch)
# train_strat = DerivativeTraining()

ds_path = "./data"
chk_path = "./chk"
eval_path = "./eval"


# with DerivativeTraining

mgn, losses_dict = train_network(
    opt_derivative, ds_path, chk_path; mps = message_steps, layer_size = layer_size,
    hidden_layers = hidden_layers, activ_function = activation_function, epochs = epo, 
    steps = Int(ns), use_cuda = cuda, checkpoint = cp_derivative, norm_steps = norm_steps, 
    types_inflow = types_inflow, types_updated = types_updated, types_noisy = types_noisy, 
    noise_stddevs = noise_stddevs, solver_valid = Euler(), solver_valid_dt = 60.0, opt_scheduler=schedule,
    training_strategy = train_strat, ad = ad, train_noiseless = train_noiseless
);

tstart = 0.0f0
dt = 60.0f0
tstop = dt*360f0

# timesteps at which the mean squared error is calculated and printed during evaluation
mse_steps = vcat(collect(tstart:dt:tstop), tstop)

eval_network(
    ds_path, chk_path, eval_path, Euler(); start = tstart, stop = tstop,
    dt = dt, saves = tstart:dt:tstop, mse_steps = collect(mse_steps), mps = message_steps,
    layer_size = layer_size, hidden_layers = hidden_layers, activ_function=activation_function, use_cuda = cuda
)