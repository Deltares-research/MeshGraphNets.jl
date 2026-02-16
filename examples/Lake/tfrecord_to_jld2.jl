cd(@__DIR__)
using Pkg
Pkg.activate("..")
using TFRecord, JLD2

trajectory_length = 1001
features = Dict{String, Dict}(
    "cells" => Dict{String, Any}(
        "type" => "static",
        "shape" => [1, -1, 3],
        "dtype" => "int32"
    ),
    "edges" => Dict{String, Any}(
        "type" => "static",
        "shape" => [1, -1, 2],
        "dtype" => "int32",
    ),
    "mesh_pos" => Dict{String, Any}(
        "type" => "static",
        "shape" => [1, -1, 2],
        "dtype" => "float32"
    ),
    "node_type" => Dict{String, Any}(
        "type" => "static",
        "shape" => [1, -1, 1],
        "dtype" => "int32"
    ),
    "velocity" => Dict{String, Any}(
        "type" => "dynamic",
        "shape" => [trajectory_length, -1, 2],
        "dtype" => "float32"
    ),
    "waterlevel" => Dict{String, Any}(
        "type" => "dynamic",
        "shape" => [trajectory_length, -1, 1],
        "dtype" => "float32"
    ),
    "bathymetry" => Dict{String, Any}(
        "type" => "static",
        "shape" => [1, -1, 1],
        "dtype" => "float32"
    )
)

function parse_data(data::TFRecord.Example)
    out = Dict{String, AbstractArray}()
    for (key, value) in features
        if key in keys(data.features.feature)
            d = reinterpret(getfield(Base, Symbol(uppercasefirst(value["dtype"]))),
                data.features.feature[key].kind.value.value[])
            dims = Tuple(reverse(replace(
                value["shape"], -1 => abs(reduce(div, value["shape"]; init = length(d))))))
            d = reshape(d, dims)
            if value["type"] == "static"
                d = repeat(d, 1, 1, trajectory_length)
            end
            out[key] = d
        end
    end
    return out
end

for file in ["train", "valid", "test"]
    i = 1
    jld2_file = jldopen("data/$file.jld2", "w")
    for traj in TFRecord.read("data/$file.tfrecord")
        traj_dict = parse_data(traj)
        traj_group = JLD2.Group(jld2_file, "trajectory_$i")

        if "cells" in keys(traj_dict)
            cells = traj_dict["cells"]
            traj_group["cells"] = cells[:, :, 1]
        elseif "edges" in keys(traj_dict)
            edges = traj_dict["edges"]
            traj_group["edges"] = edges[:,:,1]
        else
            @err "Neither cells nor edges found in file data/$file.tfrecord"
        end

        mesh_pos = traj_dict["mesh_pos"]
        node_type = traj_dict["node_type"]
        bathymetry = traj_dict["bathymetry"]
        waterlevel = traj_dict["waterlevel"]
        velocity = traj_dict["velocity"]

        for idx in axes(mesh_pos, 2)
            traj_group["node[$idx].mesh_pos"] = mesh_pos[:, idx, 1]
            traj_group["node[$idx].node_type"] = node_type[:, idx, 1]
            traj_group["node[$idx].d"] = bathymetry[:, idx, :]
            traj_group["node[$idx].h"] = waterlevel[:, idx, :]
            traj_group["node[$idx].V"] = velocity[:, idx, :]
        end
        traj_group["n_nodes"] = size(mesh_pos, 2)
        i += 1
    end
    close(jld2_file)
end