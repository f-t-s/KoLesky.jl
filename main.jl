using KoLesky
using Plots

x = rand(2, 100000) 

# x[:, 1] .= x[:, 2]

# maximin ordering for a single neighbor
@time P, ℓ = KoLesky.maximin_ordering(x, 3)


supernodes = KoLesky.supernodal_reverse_maximin_sparsity_pattern(x, P, ℓ, 2.0)

x = x[:, P]

# # Plotting the ordeing 
# outplot = scatter(xlims=(-0.1,1.1), ylims=(-0.1,1.1))
# for (k, size) in zip(reverse([50, 200, 500, 2000]), 1 : 4)
#     scatter!(outplot, x[1, P[1 : k]], x[2, P[1 : k]], markersize=size)
# end
# display(outplot)

m = KoLesky.point_measurements(x)

supernodal_assignment, P = KoLesky.supernodal_reverse_maximin(m, 3, 3.0)


# Plototing the supernodes
outplot = scatter(xlims=(-0.1,1.1), ylims=(-0.1,1.1),aspect_ratio=:equal,)
for k in [1331]
    node = supernodes[k]
    scatter!(outplot, x[1, node.column_indices], x[2, node.column_indices], markersize=5)
    scatter!(outplot, x[1, node.row_indices], x[2, node.row_indices])
end
display(outplot)