include("volumerhs.jl")

const PROFILING = haskey(ENV, "NSYS_PROFILING_SESSION_ID") || haskey(ENV, "PROFILING")

using TypedTables
using CSV

# XXX: Add to GPUArrays
Base.one(A::CuArray) = CUDA.ones(eltype(A), size(A)...)

function main(N=4, nmoist=0, ntrace=0)
    results = Table(kind=Symbol[], nelems=Int[], N=Int[], time=Float64[])
    ntrials = 10
    DFloat = Float32

    points = (1:12) * 20_000
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

    kernel = @cuda launch=false volumerhs!(rhs, Q, vgeo, DFloat(grav), D, Val(N), Val(nmoist), Val(ntrace))
    kernel(rhs, Q, vgeo, DFloat(grav), D, nelem; threads=threads, blocks=nelem)

    CUDA.@profile for _ = 1:ntrials
        res = CUDA.@timed kernel(rhs, Q, vgeo, DFloat(grav), D, Val(N), Val(nmoist), Val(ntrace); threads=threads, blocks=nelem)
        push!(results, (kind=:primal, nelems=nelem, N=N, time=res.time))
    end

    if !PROFILING
        open("vrhs.ll", "w") do io
            CUDA.@device_code_llvm io=io dump_module=true raw=true debuginfo=:none @cuda threads=threads blocks=nelem volumerhs!(rhs, Q, vgeo, DFloat(grav), D, Val(N), Val(nmoist), Val(ntrace))
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

    kernel = @cuda launch=false dvolumerhs!(rhs, Q, vgeo, DFloat(grav), D, Val(N), Val(nmoist), Val(ntrace))
    kernel(rhs, Q, vgeo, DFloat(grav), D, nelem; threads=threads, blocks=nelem)

    CUDA.@profile for _ = 1:ntrials
        res = CUDA.@timed kernel(rhs, Q, vgeo, DFloat(grav), D, Val(N), Val(nmoist), Val(ntrace); threads=threads, blocks=nelem)
        push!(results, (kind=:all_const, nelems=nelem, N=N, time=res.time))
    end

    if !PROFILING
        open("const_dvrhs.ll", "w") do io
            CUDA.@device_code_llvm io=io dump_module=true raw=true debuginfo=:none begin
                @cuda threads=threads blocks=nelem dvolumerhs_const!(rhs, Q, vgeo, DFloat(grav), D, Val(N), Val(nmoist), Val(ntrace))
            end
        end
    end

    @info "Full Enzyme run"

    drhs  = Duplicated(rhs,  one(rhs))
    dQ    = Duplicated(Q,    zero(Q))
    dvgeo = Duplicated(vgeo, zero(vgeo))
    dD    = Duplicated(D,    zero(D))

    kernel = @cuda launch=false dvolumerhs!(drhs, dQ, dvgeo, DFloat(grav), dD, Val(N), Val(nmoist), Val(ntrace))
    kernel(drhs, dQ, dvgeo, DFloat(grav), dD, Val(N), Val(nmoist), Val(ntrace) ; threads=threads, blocks=nelem)

    CUDA.@profile for _ = 1:ntrials
        res = CUDA.@timed kernel(drhs, dQ, dvgeo, DFloat(grav), dD, Val(N), Val(nmoist), Val(ntrace) ; threads=threads, blocks=nelem)
        push!(results, (kind=:all_dub, nelems=nelem, N=N, time=res.time))
    end

    if !PROFILING
        open("dvrhs.ll", "w") do io
            CUDA.@device_code_llvm io=io dump_module=true raw=true debuginfo=:none begin
                @cuda threads=threads blocks=nelem dvolumerhs!(drhs, dQ,  dvgeo, DFloat(grav), dD, Val(N), Val(nmoist), Val(ntrace))
            end
        end
    end

    CUDA.unsafe_free!(dD.dval)
    CUDA.unsafe_free!(D)
    CUDA.unsafe_free!(dQ.dval)
    CUDA.unsafe_free!(Q)
    CUDA.unsafe_free!(dvgeo.dval)
    CUDA.unsafe_free!(vgeo)
    CUDA.unsafe_free!(drhs.dval)
    CUDA.unsafe_free!(rhs)
    @show results
    end
    CSV.write("profile_$N.csv", results)
end

main()
