include("volumerhs.jl")

const PROFILING = haskey(ENV, "NSYS_PROFILING_SESSION_ID") || haskey(ENV, "PROFILING")

using TypedTables
using CSV

function main()
    results = Table(kind=Symbol[], nelems=Int[], time=Float64[])
    ntrials = 10
    DFloat = Float32

    points = [20_000, 40_000, 80_000, 160_000, 240_000]
    points = (1:12) * 20_000
    # points = 240_000

    for nelem in points

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

    kernel = @cuda launch=false volumerhs!(rhs, Q, vgeo, DFloat(grav), D, nelem)
    kernel(rhs, Q, vgeo, DFloat(grav), D, nelem; threads=threads, blocks=nelem)

    # @show rhs′ ≈ rhs

    CUDA.@profile for _ = 1:ntrials
        res = CUDA.@timed kernel(rhs, Q, vgeo, DFloat(grav), D, nelem; threads=threads, blocks=nelem)
        push!(results, (kind=:primal, nelems=nelem, time=res.time))
    end

    if !PROFILING
        open("vrhs.ll", "w") do io
            CUDA.@device_code_llvm io=io dump_module=true raw=true debuginfo=:none @cuda threads=threads blocks=nelem volumerhs!(rhs, Q, vgeo, DFloat(grav), D, nelem)
        end
    end

    @info "Starting Enzyme run"

    if !PROFILING
        # Enzyme.API.EnzymeSetCLBool(:EnzymePreopt, false)
        Enzyme.API.EnzymeSetCLBool(:EnzymePrintPerf, true)
        Enzyme.API.EnzymeSetCLBool(:EnzymePrint, true)
        # Enzyme.API.EnzymeSetCLBool(:EnzymePrintActivity, true)
    end

    Enzyme.API.EnzymeSetCLBool(:EnzymeRegisterReduce, false)
    # Enzyme.API.EnzymeSetCLString(:EnzymeBCPath, "/home/wmoses/git/Enzyme/enzyme/bclib")

    kernel = @cuda launch=false dvolumerhs_const!(rhs, Q, vgeo, DFloat(grav), D, nelem)
    kernel(rhs, Q, vgeo, DFloat(grav), D, nelem; threads=threads, blocks=nelem)

    CUDA.@profile for _ = 1:ntrials
        res = CUDA.@timed kernel(rhs, Q, vgeo, DFloat(grav), D, nelem; threads=threads, blocks=nelem)
        push!(results, (kind=:all_const, nelems=nelem, time=res.time))
    end

    if !PROFILING
        open("const_dvrhs.ll", "w") do io
            CUDA.@device_code_llvm io=io dump_module=true raw=true debuginfo=:none @cuda threads=threads blocks=nelem dvolumerhs_const!(rhs, Q, vgeo, DFloat(grav), D, nelem)
        end
    end

    @info "Full Enzyme run"

    drhs = similar(rhs)
    drhs .= 1

    dvgeo = similar(vgeo)
    dvgeo .= 0

    dQ = similar(Q)
    dQ .= 0

    dD = similar(D)
    dD .= 0

    kernel = @cuda launch=false dvolumerhs!(rhs, drhs, Q, dQ, vgeo, dvgeo, DFloat(grav), D, dD, nelem)
    kernel(rhs, drhs, Q, dQ, vgeo, dvgeo, DFloat(grav), D, dD, nelem; threads=threads, blocks=nelem)

    CUDA.@profile for _ = 1:ntrials
        res = CUDA.@timed kernel(rhs, drhs, Q, dQ, vgeo, dvgeo, DFloat(grav), D, dD, nelem; threads=threads, blocks=nelem)
        push!(results, (kind=:all_dub, nelems=nelem, time=res.time))
    end

    if !PROFILING
        open("dvrhs.ll", "w") do io
            CUDA.@device_code_llvm io=io dump_module=true raw=true debuginfo=:none @cuda threads=threads blocks=nelem dvolumerhs!(rhs, drhs, Q, dQ, vgeo, dvgeo, DFloat(grav), D, dD, nelem)
        end
    end

    CUDA.unsafe_free!(dD)
    CUDA.unsafe_free!(D)
    CUDA.unsafe_free!(dQ)
    CUDA.unsafe_free!(Q)
    CUDA.unsafe_free!(dvgeo)
    CUDA.unsafe_free!(vgeo)
    CUDA.unsafe_free!(drhs)
    CUDA.unsafe_free!(rhs)
    @show results
    end
    CSV.write("profile.csv", results)
end

main()
