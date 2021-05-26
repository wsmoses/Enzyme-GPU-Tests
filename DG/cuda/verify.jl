include("volumerhs.jl")

function main()
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
    # rhs′ = CuArray(zeros(DFloat, Nq, Nq, Nq, nvar, nelem))

    # CUDA.limit!(CUDA.CU_LIMIT_MALLOC_HEAP_SIZE, 1*1024^3)
    # CUDA.cache_config!(CUDA.CU_FUNC_CACHE_PREFER_L1)

    threads=(N+1, N+1)

    # @cuda threads=threads blocks=nelem volumerhs_good!(rhs′, Q, vgeo, DFloat(grav), D, nelem)

    @info "Starting Enzyme run"

    Enzyme.API.EnzymeSetCLBool(:EnzymeRegisterReduce, true)
    # Enzyme.API.EnzymeSetCLString(:EnzymeBCPath, "/home/wmoses/git/Enzyme/enzyme/bclib")

    drhs = similar(rhs)
    drhs .= 0
    drhs[1, 1, 1, 2, 1:1] .= 1

    dvgeo = similar(vgeo)
    dvgeo .= 0

    dQ = similar(Q)
    dQ .= 0

    dD = similar(D)
    dD .= 0

    kernel = @cuda launch=false dvolumerhs!(rhs, drhs, Q, dQ, vgeo, dvgeo, DFloat(grav), D, dD, nelem)
    kernel(rhs, drhs, Q, dQ, vgeo, dvgeo, DFloat(grav), D, dD, nelem; threads=threads, blocks=nelem)

    o1 = rhs[1, 1, 1, 2, 1:1]
    Q[1] += 1e-4
    rhs .= 0

    kernel = @cuda launch=false volumerhs!(rhs, Q, vgeo, DFloat(grav), D, nelem)
    kernel(rhs, Q, vgeo, DFloat(grav), D, nelem; threads=threads, blocks=nelem)
    o2 = rhs[1, 1, 1, 2, 1:1]
    @show dQ[1], (o2-o1) / 1e-4
end

main()
