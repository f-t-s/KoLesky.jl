using KoLesky
using Plots
using Random
using LinearAlgebra

Random.seed!(123)

N = 10000
x = rand(2, N) 


# x[:, 1] .= x[:, 2]

# maximin ordering for a single neighbor
# @time P, ℓ = KoLesky.maximin_ordering(x, 3)
# 
# 
# supernodes = KoLesky.supernodal_reverse_maximin_sparsity_pattern(x, P, ℓ, 2.0)
# 
# x = x[:, P]

# Plotting the ordeing 
# outplot = scatter(xlims=(-0.1,1.1), ylims=(-0.1,1.1))
# for (k, size) in zip(reverse([50, 200, 500, 2000]), 1 : 4)
#     scatter!(outplot, x[1, P[1 : k]], x[2, P[1 : k]], markersize=size)
# end
# display(outplot)


# P, ℓ, supernodes =  KoLesky.ordering_and_sparsity_pattern([rand(3, 10), rand(3, 9)], 3, Inf)


# reconstructing problem case
# x = [0.7011248193214994 0.5287784291388624 0.9912073930216181 0.011022708514755397 0.3048911800120049 0.7633167461586408 0.148450446160298 0.4998541695980816 0.9877685201303661; 0.7417671385050895 0.9344584852405626 0.6751856871300057 0.17238567999833831 0.1870493702497309 0.08482050061309887 0.5341260387882831 0.8036510878409324 0.7561844091125525; 0.9870715010828004 0.9898603530398478 0.05180842461513291 0.12403603428784793 0.06708617251780202 0.409731150087008 0.03035237262281698 0.6732108974166173 0.2401251248066787]
# 
# init_distances = [0.701535793155449 0.6476581460633896 0.4957260187085011 0.8022366352509899 0.6479527352435571 0.39172328870039463 0.6427357124884787 0.4866490592991009 0.47345627514182653; 0.3867659061378994 0.41309337149730185 0.3554033819884916 0.7681219055679981 0.5159431172310169 0.25695776446634716 0.5880807749655886 0.3575476193522463 0.471778887527783; 0.23581678996822433 0.3650869188729263 0.3547573477611862 0.7132807419319691 0.41791845952335266 0.19490775985230116 0.42958795026324476 0.027678044757786485 0.3422232129738419]
# 
# # init_distances = fill(Inf, 3, size(x, 2))
# 
# 
# P, ℓ =  KoLesky.maximin_ordering(x, 3; init_distances)

measurements = KoLesky.point_measurements(x)

𝒢 = KoLesky.MaternCovariance1_2(0.1)

implicit_factor = KoLesky.ImplicitKLFactorization(𝒢, measurements, 12.0)

@time explicit_factor = KoLesky.ExplicitKLFactorization(implicit_factor)

# comparing to true result

KM = zeros(N, N)
𝒢(KM, measurements)

@show norm(KoLesky.assemble_covariance(explicit_factor) - KM ) / norm(KM)