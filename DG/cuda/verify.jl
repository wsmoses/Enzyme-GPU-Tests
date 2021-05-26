include("volumerhs.jl")

# XXX: Add to GPUArrays
Base.one(A::CuArray) = CUDA.ones(eltype(A), size(A)...)

function main(N=4, nmoist=0, ntrace=0)
    DFloat = Float32

    nelem = 20_000

    rnd = MersenneTwister(0)

    Nq = N + 1
    nvar = _nstate + nmoist + ntrace

    Q = 1 .+ CuArray(rand(rnd, DFloat, Nq, Nq, Nq, nvar, nelem))
    Q[:, :, :, _E, :] .+= 20
    vgeo = CuArray(rand(rnd, DFloat, Nq, Nq, Nq, _nvgeo, nelem))

    # Make sure the entries of the mass matrix satisfy the inverse relation
    vgeo[:, :, :, _MJ, :] .+= 3
    vgeo[:, :, :, _MJI, :] .= 1 ./ vgeo[:, :, :, _MJ, :]

    D = CuArray(rand(rnd, DFloat, Nq, Nq))

    rhs = CuArray(zeros(DFloat, Nq, Nq, Nq, nvar, nelem))

    # CUDA.limit!(CUDA.CU_LIMIT_MALLOC_HEAP_SIZE, 1*1024^3)
    # CUDA.cache_config!(CUDA.CU_FUNC_CACHE_PREFER_L1)

    threads=(N+1, N+1)

    @info "Starting Enzyme run"

    Enzyme.API.EnzymeSetCLBool(:EnzymeRegisterReduce, true)
    # Enzyme.API.EnzymeSetCLString(:EnzymeBCPath, "/home/wmoses/git/Enzyme/enzyme/bclib")

    drhs  = Duplicated(rhs,  zero(rhs))
    drhs.dval[1, 1, 1, 2, 1:1] .= 1
    dQ    = Duplicated(Q,    zero(Q))
    dvgeo = Duplicated(vgeo, zero(vgeo))
    dD    = Duplicated(D,    zero(D))

    @cuda dvolumerhs!(drhs, dQ, dvgeo, DFloat(grav), dD, Val(N), Val(nmoist), Val(ntrace))

    o1 = rhs[1, 1, 1, 2, 1:1]
    Q[1] += 1e-4
    rhs .= 0

    @cuda volumerhs!(rhs, Q, vgeo, DFloat(grav), D, Val(N), Val(nmoist), Val(ntrace))
    o2 = rhs[1, 1, 1, 2, 1:1]
    @show dQ.dval[1], (o2-o1) / 1e-4
end

main()
