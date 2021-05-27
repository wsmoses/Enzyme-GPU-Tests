using AMDGPU
using StaticArrays
using Random
using Enzyme
using LLVM
using LLVM.Interop

include(joinpath(@__DIR__, "..", "utils.jl"))

const PROFILING = haskey(ENV, "PROFILING")

# note the order of the fields below is also assumed in the code.
const _nstate = 5
const _ρ, _U, _V, _W, _E = 1:_nstate
const stateid = (ρ = _ρ, U = _U, V = _V, W = _W, E = _E)

const _nvgeo = 14
const _ξx, _ηx, _ζx, _ξy, _ηy, _ζy, _ξz, _ηz, _ζz, _MJ, _MJI,
_x, _y, _z = 1:_nvgeo
const vgeoid = (ξx = _ξx, ηx = _ηx, ζx = _ζx,
                ξy = _ξy, ηy = _ηy, ζy = _ζy,
                ξz = _ξz, ηz = _ηz, ζz = _ζz,
                MJ = _MJ, MJI = _MJI,
                x = _x,   y = _y,   z = _z)

# TODO:
# Enzyme can't handle functions with `Val` inputs
const N = 4
const nmoist = 0
const ntrace = 0

Base.@irrational grav 9.81 BigFloat(9.81)
Base.@irrational gdm1 0.4 BigFloat(0.4)

@generated function addrspacecast(::Type{to}, from) where {to}
   JuliaContext() do ctx
        # get the element type
        to_typ = convert(LLVMType, to, ctx)
        from_typ = convert(LLVMType, from, ctx)

        paramtyps = [from_typ]
        llvmf, _ = create_function(to_typ, paramtyps)

        # generate IR
        Builder(ctx) do builder
            entry = BasicBlock(llvmf, "entry", ctx)
            position!(builder, entry)
            val = addrspacecast!(builder, parameters(llvmf)[1], to_typ)
            ret!(builder, val)
        end

        call_function(llvmf, to, Tuple{from}, :(from,))
    end
end

# TODO: Add to AMDGPU.jl
function Base.Experimental.Const(arr::AMDGPU.ROCDeviceArray{T,N, AMDGPU.AS.Global}) where {T, N}
    ptr = arr.ptr
    ptr = addrspacecast(Core.LLVMPtr{T, AMDGPU.AS.Constant}, ptr)
    AMDGPU.ROCDeviceArray(arr.shape, ptr)
end

function volumerhs!(rhs, Q, vgeo, gravity, D, nelem)
    Nq = N + 1
    nvar = _nstate + nmoist + ntrace

    rhs = PartialKnownDims{Tuple{Nq, Nq, Nq, nvar, Unkown()}}(rhs)
 
    Q = PartialKnownDims{Tuple{Nq, Nq, Nq, nvar, Unkown()}}(Q)
    Q = Base.Experimental.Const(Q)

    vgeo = PartialKnownDims{Tuple{Nq, Nq, Nq, _nvgeo, Unkown()}}(vgeo)
    vgeo = Base.Experimental.Const(vgeo)

    D = PartialKnownDims{Tuple{Nq, Nq}}(D)
    D = Base.Experimental.Const(D)

    s_D = let ptr = AMDGPU.alloc_special(Val(:s_D), eltype(D), Val(AMDGPU.AS.Local), Val(prod((Nq, Nq))))
        AMDGPU.ROCDeviceArray((Nq, Nq), ptr)
    end

    s_F = let ptr = AMDGPU.alloc_special(Val(:s_F), eltype(D), Val(AMDGPU.AS.Local), Val(prod((Nq, Nq, _nstate))))
        AMDGPU.ROCDeviceArray((Nq, Nq, _nstate), ptr)
    end

    s_G = let ptr = AMDGPU.alloc_special(Val(:s_G), eltype(D), Val(AMDGPU.AS.Local), Val(prod((Nq, Nq, _nstate))))
        AMDGPU.ROCDeviceArray((Nq, Nq, _nstate), ptr)
    end

    r_rhsρ = MArray{Tuple{Nq}, eltype(rhs)}(undef)
    r_rhsU = MArray{Tuple{Nq}, eltype(rhs)}(undef)
    r_rhsV = MArray{Tuple{Nq}, eltype(rhs)}(undef)
    r_rhsW = MArray{Tuple{Nq}, eltype(rhs)}(undef)
    r_rhsE = MArray{Tuple{Nq}, eltype(rhs)}(undef)

    e = workgroupIdx().x
    j = workitemIdx().y
    i = workitemIdx().x

    @inbounds begin
        for k in 1:Nq
            r_rhsρ[k] = zero(eltype(rhs))
            r_rhsU[k] = zero(eltype(rhs))
            r_rhsV[k] = zero(eltype(rhs))
            r_rhsW[k] = zero(eltype(rhs))
            r_rhsE[k] = zero(eltype(rhs))
        end

        # fetch D into shared
        s_D[i, j] = D[i, j]
        # let k = 1
        @unroll for k in 1:Nq
            sync_workgroup()

            # Load values will need into registers
            MJ = vgeo[i, j, k, _MJ, e]
            ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
            ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
            ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]
            z = vgeo[i,j,k,_z,e]

            U, V, W = Q[i, j, k, _U, e], Q[i, j, k, _V, e], Q[i, j, k, _W, e]
            ρ, E = Q[i, j, k, _ρ, e], Q[i, j, k, _E, e]
            
            # GPU performance trick
            # Allow optimizations to use the reciprocal of an argument rather than perform division.
            # IEEE floating-point division is implemented as a function call
            ρinv = rcp(ρ)
            ρ2inv = rcp(2ρ)
            # ρ2inv = 0.5f0 * pinv

            P = gdm1*(E - (U^2 + V^2 + W^2)*ρ2inv - ρ*gravity*z)

            fluxρ_x = U
            fluxU_x = ρinv * U * U + P
            fluxV_x = ρinv * U * V
            fluxW_x = ρinv * U * W
            fluxE_x = ρinv * U * (E + P)

            fluxρ_y = V
            fluxU_y = ρinv * V * U
            fluxV_y = ρinv * V * V + P
            fluxW_y = ρinv * V * W
            fluxE_y = ρinv * V * (E + P)

            fluxρ_z = W
            fluxU_z = ρinv * W * U
            fluxV_z = ρinv * W * V
            fluxW_z = ρinv * W * W + P
            fluxE_z = ρinv * W * (E + P)

            s_F[i, j,  _ρ] = MJ * (ξx * fluxρ_x + ξy * fluxρ_y + ξz * fluxρ_z)
            s_F[i, j,  _U] = MJ * (ξx * fluxU_x + ξy * fluxU_y + ξz * fluxU_z)
            s_F[i, j,  _V] = MJ * (ξx * fluxV_x + ξy * fluxV_y + ξz * fluxV_z)
            s_F[i, j,  _W] = MJ * (ξx * fluxW_x + ξy * fluxW_y + ξz * fluxW_z)
            s_F[i, j,  _E] = MJ * (ξx * fluxE_x + ξy * fluxE_y + ξz * fluxE_z)

            s_G[i, j,  _ρ] = MJ * (ηx * fluxρ_x + ηy * fluxρ_y + ηz * fluxρ_z)
            s_G[i, j,  _U] = MJ * (ηx * fluxU_x + ηy * fluxU_y + ηz * fluxU_z)
            s_G[i, j,  _V] = MJ * (ηx * fluxV_x + ηy * fluxV_y + ηz * fluxV_z)
            s_G[i, j,  _W] = MJ * (ηx * fluxW_x + ηy * fluxW_y + ηz * fluxW_z)
            s_G[i, j,  _E] = MJ * (ηx * fluxE_x + ηy * fluxE_y + ηz * fluxE_z)

            r_Hρ = MJ * (ζx * fluxρ_x + ζy * fluxρ_y + ζz * fluxρ_z)
            r_HU = MJ * (ζx * fluxU_x + ζy * fluxU_y + ζz * fluxU_z)
            r_HV = MJ * (ζx * fluxV_x + ζy * fluxV_y + ζz * fluxV_z)
            r_HW = MJ * (ζx * fluxW_x + ζy * fluxW_y + ζz * fluxW_z)
            r_HE = MJ * (ζx * fluxE_x + ζy * fluxE_y + ζz * fluxE_z)

            # one shared access per 10 flops
            for n = 1:Nq
                Dkn = s_D[k, n]

                r_rhsρ[n] += Dkn * r_Hρ
                r_rhsU[n] += Dkn * r_HU
                r_rhsV[n] += Dkn * r_HV
                r_rhsW[n] += Dkn * r_HW
                r_rhsE[n] += Dkn * r_HE
            end

            r_rhsW[k] -= MJ * ρ * gravity

            sync_workgroup()

            # loop of ξ-grid lines
            @unroll for n = 1:Nq
                Dni = s_D[n, i]
                Dnj = s_D[n, j]

                r_rhsρ[k] += Dni * s_F[n, j, _ρ]
                r_rhsρ[k] += Dnj * s_G[i, n, _ρ]

                r_rhsU[k] += Dni * s_F[n, j, _U]
                r_rhsU[k] += Dnj * s_G[i, n, _U]

                r_rhsV[k] += Dni * s_F[n, j, _V]
                r_rhsV[k] += Dnj * s_G[i, n, _V]

                r_rhsW[k] += Dni * s_F[n, j, _W]
                r_rhsW[k] += Dnj * s_G[i, n, _W]

                r_rhsE[k] += Dni * s_F[n, j, _E]
                r_rhsE[k] += Dnj * s_G[i, n, _E]
            end 
        end # k

        @unroll for k in 1:Nq
            MJI = vgeo[i, j, k, _MJI, e]

            # Updates are a performance bottleneck
            # Check if coalesced properly
            # primary source of stall_long_sb
            rhs[i, j, k, _U, e] += MJI*r_rhsU[k]
            rhs[i, j, k, _V, e] += MJI*r_rhsV[k]
            rhs[i, j, k, _W, e] += MJI*r_rhsW[k]
            rhs[i, j, k, _ρ, e] += MJI*r_rhsρ[k]
            rhs[i, j, k, _E, e] += MJI*r_rhsE[k]
        end
    end
    return nothing
end

function dvolumerhs!(rhs, drhs, Q, dQ, vgeo, dvgeo, grav, D, dD, nelem)
    Enzyme.autodiff_no_cassette(volumerhs!, Duplicated(rhs, drhs), Duplicated(Q,dQ), Duplicated(vgeo, dvgeo), grav, Duplicated(D, dD), nelem)
    return nothing
end
function dvolumerhs_const!(rhs, Q, vgeo, grav, D, nelem)
    Enzyme.autodiff_no_cassette(volumerhs!, rhs, Q, vgeo, grav, D, nelem)
    return nothing
end

using TypedTables
using CSV

function main()
    results = Table(kind=Symbol[], nelems=Int[], time=Float64[])    
    ntrials = 10
    DFloat = Float32

    # points = [20_000, 40_000, 80_000, 160_000]
    points = 240_000
    for nelem in points
    GC.gc()
    GC.gc()

    rnd = MersenneTwister(0)

    Nq = N + 1
    nvar = _nstate + nmoist + ntrace

    Q = 1 .+ rand(rnd, DFloat, Nq, Nq, Nq, nvar, nelem)
    Q[:, :, :, _E, :] .+= 20
    Q = ROCArray(Q)
    vgeo = rand(rnd, DFloat, Nq, Nq, Nq, _nvgeo, nelem)

    # Make sure the entries of the mass matrix satisfy the inverse relation
    vgeo[:, :, :, _MJ, :] .+= 3
    vgeo[:, :, :, _MJI, :] .= 1 ./ vgeo[:, :, :, _MJ, :]
    vgeo = ROCArray(vgeo)

    D = ROCArray(rand(rnd, DFloat, Nq, Nq))

    rhs = ROCArray(zeros(DFloat, Nq, Nq, Nq, nvar, nelem))

    # CUDA.limit!(CUDA.CU_LIMIT_MALLOC_HEAP_SIZE, 1*1024^3)
    # CUDA.cache_config!(CUDA.CU_FUNC_CACHE_PREFER_L1)

    threads=(N+1, N+1)
    AMDGPU.@device_code dir="vrhs" @roc groupsize=threads gridsize=nelem volumerhs!(rhs, Q, vgeo, DFloat(grav), D, nelem)

    if !PROFILING
        open("vrhs.ll", "w") do io
            AMDGPU.@device_code_llvm io=io dump_module=true raw=true debuginfo=:none @roc groupsize=threads gridsize=nelem volumerhs!(rhs, Q, vgeo, DFloat(grav), D, nelem)
        end
    end

    kernel = @roc launch=false volumerhs!(rhs, Q, vgeo, DFloat(grav), D, nelem)
    signal = AMDGPU.create_event(kernel.mod.exe)
    kernel(rhs, Q, vgeo, DFloat(grav), D, nelem; groupsize=threads, gridsize=nelem, signal=signal)
    wait(signal)

    for _ = 1:ntrials
    	signal = AMDGPU.create_event(kernel.mod.exe)
    	res = @timed begin
    kernel(rhs, Q, vgeo, DFloat(grav), D, nelem; groupsize=threads, gridsize=nelem, signal=signal)
    wait(signal, minlat=0.001)
    	end
	push!(results, (kind=:primal, nelems=nelem, time=res.time))
    end

    @info "Starting Enzyme run"
    flush(stderr)
    flush(stdout)
    if !PROFILING
        # Enzyme.API.EnzymeSetCLBool(:EnzymePreopt, false)
        Enzyme.API.EnzymeSetCLBool(:EnzymePrintPerf, true)
        Enzyme.API.EnzymeSetCLBool(:EnzymePrint, true)
    end

    kernel = @roc launch=false dvolumerhs_const!(rhs, Q, vgeo, DFloat(grav), D, nelem)
    signal = AMDGPU.create_event(kernel.mod.exe)
    kernel(rhs, Q, vgeo, DFloat(grav), D, nelem; groupsize=threads, gridsize=nelem, signal=signal)
    wait(signal)

    for _ = 1:ntrials
	    	signal = AMDGPU.create_event(kernel.mod.exe)
	res = @timed begin
	    	kernel(rhs, Q, vgeo, DFloat(grav), D, nelem; groupsize=threads, gridsize=nelem, signal=signal)
    		wait(signal, minlat=0.001)
	end
	push!(results, (kind=:all_const, nelems=nelem, time=res.time))
    end

    if !PROFILING
        open("const_dvrhs.ll", "w") do io
            AMDGPU.@device_code_llvm io=io dump_module=true raw=true debuginfo=:none @roc groupsize=threads gridsize=nelem dvolumerhs_const!(rhs, Q, vgeo, DFloat(grav), D, nelem)
        end
    end

    @info "Full Enzyme run"
    flush(stderr)

    drhs = similar(rhs)
    drhs .= 1

    dvgeo = similar(vgeo)
    dvgeo .= 0

    dQ = similar(Q)
    dQ .= 0

    dD = similar(D)
    dD .= 0


    kernel = @roc launch=false dvolumerhs!(rhs, drhs, Q, dQ, vgeo, dvgeo, DFloat(grav), D, dD, nelem)
    signal = AMDGPU.create_event(kernel.mod.exe)
    kernel(rhs, drhs, Q, dQ, vgeo, dvgeo, DFloat(grav), D, dD, nelem; groupsize=threads, gridsize=nelem, signal=signal)
    wait(signal)

    for _ = 1:ntrials
    		signal = AMDGPU.create_event(kernel.mod.exe)
	res = @timed begin
    		kernel(rhs, drhs, Q, dQ, vgeo, dvgeo, DFloat(grav), D, dD, nelem; groupsize=threads, gridsize=nelem, signal=signal)
    		wait(signal, minlat=0.001)
    	end
	push!(results, (kind=:all_dub, nelems=nelem, time=res.time))
    end

    if !PROFILING
        open("dvrhs.ll", "w") do io
            AMDGPU.@device_code_llvm io=io dump_module=true raw=true debuginfo=:none @roc groupsize=threads gridsize=nelem dvolumerhs!(rhs, drhs, Q, dQ, vgeo, dvgeo, DFloat(grav), D, dD, nelem)
        end
    end
    @show results
    end
    CSV.write("profile.csv", results)
end

main()
