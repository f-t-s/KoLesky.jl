module KoLesky

# Write your package code here.

include("./DOF.jl")
include("./Sorting.jl")
include("./SuperNodes.jl")

export mat2points
export maximin
export partition_into_supernodes


end
