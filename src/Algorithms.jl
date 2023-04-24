import Base.rand
############################################################
# Computing prediction
############################################################
#Assumes that the values of y are already written into the buffer.
#They will be overwritten during the execution of the function
function predictSkeleton!( μ, σ, L, B, y::AbstractVector, α, β, γ, δ, ℓ, skel, Kij!, ITest )
    NTest = length(ITest)
    NTrain = length(skel.children)
    # We are using reverse ordering during this function
    reverse!(skel.parents)
    reverse!(skel.children)
    reverse!(y)

    Kij!(L, skel.children, skel.children)

    # Compute the cholesky factor of the sub-covariance matrix
    L = cholesky!(Hermitian(L)).L

    # Compute B
    Kij!(B, skel.children, ITest)

    # Compute the intermediate quantities that use inner products for the 
    # First value of k. 
    # They will be updated by substracting the corresponding columns for the different
    # k
    ldiv!(L, B)
    ldiv!(L, y)
    # Should not be necessary but presently mapreduce seems to allocate excessive amounts of memory
    for j = 1 : NTest
        for i = 1 : NTrain
            α[j] += y[i] * B[i,j]
            β[j] += B[i,j]^2
        end
    end

    old_k = length(skel.children) 
    for k_par = length(skel.parents) : -1 : 1 
        k = findlast(x -> x == skel.parents[k_par],
            skel.children)

        for l = (k + 1) : (old_k)
            for j = 1 : NTest
                α[j] -= y[l] * B[l,j] 
                β[j] -= B[l,j]^2
            end
        end
        for j = 1 : NTest
            γ[j] = sqrt(one(eltype(γ)) + (B[k,j])^2 / (δ[j] - β[j]))
        end

        for j = 1 : NTest
            ℓ[j] = - B[k,j] * (one(eltype(ℓ)) + β[j] / (δ[j] - β[j])) / δ[j] / γ[j] 
        end

        for j = 1 : NTest
            μ[j] += ℓ[j] / γ[j] * (y[k] + B[k,j] * α[j] / (δ[j] - β[j]))
        end
        @. σ += ℓ^2
        old_k = k
    end

    # We are reverting the reversion, for now, to not interfere with parts of the
    # code that might require the original ordering.
    reverse!(skel.parents)
    reverse!(skel.children)
end

# Version for multiple samples and therefore matrix-valued y
function predictSkeleton!( μ, σ, L, B, y::AbstractMatrix, α, β, γ, δ, ℓ, skel, Kij!, ITest)
    NTest = length(ITest)
    NTrain = length(skel.children)
    NSamples = size(y, 2)
    # We are using reverse ordering during this function
    reverse!(skel.parents)
    reverse!(skel.children)
    # a little cumbersome:
    for k_sample = 1 : NSamples
        reverse!(vec(view(y, :, k_sample)))
    end

    Kij!(L, skel.children, skel.children)

    # Compute the cholesky factor of the sub-covariance matrix
    L = cholesky!(Hermitian(L)).L

    # Compute B
    Kij!(B, skel.children, ITest)

    # Compute the intermediate quantities that use inner products for the 
    # First value of k. 
    # They will be updated by substracting the corresponding columns for the different
    # k
    ldiv!(L, B)
    ldiv!(L, y)
    # Should not be necessary but presently mapreduce seems to allocate excessive amounts of memory
    # Should replace with matrix inverse
    for j = 1 : NTest
        for i = 1 : NTrain
            for k_sample = 1 : NSamples   
                α[j, k_sample] += y[i, k_sample] * B[i,j]
            end
            β[j] += B[i,j]^2
        end
    end

    old_k = length(skel.children) 
    for k_par = length(skel.parents) : -1 : 1 
        k = findlast(x -> x == skel.parents[k_par],
            skel.children)

        for l = (k + 1) : (old_k)
            for j = 1 : NTest
                for k_sample = 1 : NSamples
                    α[j, k_sample] -= y[l, k_sample] * B[l, j] 
                end
                β[j] -= B[l,j]^2
            end
        end
        for j = 1 : NTest
            γ[j] = sqrt(one(eltype(γ)) + (B[k,j])^2 / (δ[j] - β[j]))
        end

        for j = 1 : NTest
            ℓ[j] = - B[k,j] * (one(eltype(ℓ)) + β[j] / (δ[j] - β[j])) / δ[j] / γ[j] 
        end

        for j = 1 : NTest
            for k_sample = 1 : NSamples
                μ[j, k_sample] += ℓ[j] / γ[j] * (y[k, k_sample] + B[k,j] * α[j, k_sample] / (δ[j] - β[j]))
            end
        end
        @. σ += ℓ^2
        old_k = k
    end

    # We are reverting the reversion, for now, to not interfere with parts of the
    # code that might require the original ordering.
    reverse!(skel.parents)
    reverse!(skel.children)
end


# Version of predict for matrix valued data
function predict_threaded(skeletons::Vector{Skeleton{Ti}}, ITest::AbstractVector{Ti}, yTrain::AbstractMatrix, Kij!) where {Ti}
    n_threads = Threads.nthreads()
    # Computing the required size of the buffers 
    maxChildren = maximum(length.( getfield.(skeletons, :children)))
    maxParents = maximum(length.( getfield.(skeletons, :parents)))
    NTest = length(ITest)
    NTrain = sum( length.(getfield.(skeletons, :parents)))
    # We expect that yTrain has size NTrain times NSamples
    NSamples = size(yTrain, 2)

    
    #Preallocating buffers 
    LBuffer = Matrix{Float64}(undef, maxChildren^2, n_threads)
    BBuffer = Matrix{Float64}(undef, maxChildren * NTest, n_threads)
    yBuffer = Matrix{Float64}(undef, maxChildren * NSamples, n_threads)

    μs = Vector{Matrix{Float64}}(undef, n_threads)
    σs = Vector{Vector{Float64}}(undef, n_threads)
    αs = Vector{Matrix{Float64}}(undef, n_threads)
    βs = Vector{Vector{Float64}}(undef, n_threads)
    γs = Vector{Vector{Float64}}(undef, n_threads)
    ℓs = Vector{Vector{Float64}}(undef, n_threads)

    # A little cumbersome since VML.jl doesn't like subarrays
    # δ is only being read from, hence we don't need a seperate version for each thread
    δ = Vector{Float64}(undef, NTest)
    tmp = Vector{Float64}(undef, 1)
    for i = 1 : NTest
        Kij!(tmp, ITest[i:i], ITest[i:i])
        δ[i] = tmp[1]
    end

    for k = 1 : n_threads
        μs[k] = zeros(NTest, NSamples)
        σs[k] = zeros(NTest)
        γs[k] = Vector{Float64}(undef, NTest)
        αs[k] = Matrix{Float64}(undef, NTest, NSamples)
        βs[k] = Vector{Float64}(undef, NTest)
        ℓs[k] = Vector{Float64}(undef, NTest)
    end

    lin_inds_L = LinearIndices(LBuffer) 
    lin_inds_B = LinearIndices(BBuffer) 
    lin_inds_y = LinearIndices(yBuffer) 

    LinearAlgebra.BLAS.set_num_threads(1)
    GC.@preserve LBuffer BBuffer yBuffer Threads.@threads for skel in skeletons 
        nChildren = length(skel.children)
        # Setting up the outputs:
        L = unsafe_wrap(Array{Float64,2}, pointer(LBuffer, lin_inds_L[1, Threads.threadid()]), (nChildren, nChildren))
        B = unsafe_wrap(Array{Float64,2}, pointer(BBuffer, lin_inds_B[1, Threads.threadid()]), (nChildren, NTest))
        y = unsafe_wrap(Array{Float64,2}, pointer(yBuffer, lin_inds_y[1, Threads.threadid()]), (nChildren, NSamples))
        μ = μs[Threads.threadid()]
        σ = σs[Threads.threadid()]
        α = αs[Threads.threadid()]
        β = βs[Threads.threadid()]
        γ = γs[Threads.threadid()]
        ℓ = ℓs[Threads.threadid()]

        y .= yTrain[skel.children, :]
        α .= 0.0
        β .= 0.0
        γ .= 0.0
        ℓ .= 0.0

        # Do the prediction using the io variables defined above
        predictSkeleton!(μ, σ, L, B, y, α, β, γ, δ, ℓ, skel, Kij!, ITest)
    end

    μ = sum(μs)
    σ = sum(σs)

    @. σ .+= 1. / δ
    @. σ = 1. / σ
    @. μ = - σ * μ 
    return μ, σ
end


