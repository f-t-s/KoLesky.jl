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

measurements = KoLesky.point_measurements(x)

𝒢 = KoLesky.ExponentialCovariance(0.1)

implicit_factor = KoLesky.ImplicitKLFactorization(𝒢, measurements, 3.0)

explicit_factor = KoLesky.ExplicitKLFactorization(implicit_factor)