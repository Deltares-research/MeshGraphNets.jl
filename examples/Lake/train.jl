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

message_steps = 15
layer_size = 128
hidden_layers = 2
activation_function = swish
epo = 1
ns = 20
nbatch = 1
norm_steps = 0
train_noiseless = 10
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
lr_decay = 0.9f0
lr_decay_rate = 5
opt_derivative = Adam(learning_rate_derivative)
schedule = Step(start=learning_rate_derivative, decay=lr_decay, step_sizes=lr_decay_rate)

train_strat = DerivativeBatchTraining(batch_size=nbatch)
# train_strat = DerivativeTraining()

ds_path = "./data"
chk_path = "./chk_batch"
eval_path = "./eval"


# with DerivativeTraining

mgn, losses_dict = train_network(
    opt_derivative, ds_path, chk_path; mps = message_steps, layer_size = layer_size,
    hidden_layers = hidden_layers, activ_function = activation_function, epochs = epo, 
    steps = Int(ns), use_cuda = cuda, checkpoint = cp_derivative, norm_steps = norm_steps, 
    types_inflow = types_inflow, types_updated = types_updated, types_noisy = types_noisy, 
    noise_stddevs = noise_stddevs, solver_valid = Euler(), solver_valid_dt = 10000,
    training_strategy = train_strat, ad = ad, train_noiseless = train_noiseless
);