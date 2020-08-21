using KoLesky
using Plots 
import Random.seed!
seed!(123)

uptos = [1, 100, 500]

x = rand(2, 200000)

dofs = mat2points(x)

@time P_r, ℓ_r = r_maximin(dofs; n_bins=100)
@time P, ℓ = maximin(dofs)

# pl = scatter(aspect_ratio=:equal, size=(500,500))
# 
# for k = 2 : length(uptos)
#     scatter!(pl, vec(x[1, P[uptos[k-1] : uptos[k]]]), vec(x[2, P[uptos[k-1] : uptos[k]]]))
# end
# 
# display(pl)
# 