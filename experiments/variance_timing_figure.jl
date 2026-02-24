# experiments/variance_timing_figure.jl
#
# Benchmark: SMMPP_p vs matched MMTCP_{2p} for the inter-event standard
# deviation σ_T = √(m₂ − m₁²).
#
# For each p ∈ {5, 10, …, 100} we generate 200 random SMMPP_p instances,
# construct the associated MMTCP_{2p}, and record:
#   • Wall time for computing σ_T (median over a few timing runs per p).
#   • Relative difference |σ_T^SMMPP − σ_T^MMTCP| / max(σ_T^SMMPP, σ_T^MMTCP)
#     — median and 5th/95th percentiles over the 200 instances.
#
# Each random SMMPP_p is generated with off-diagonal entries of Q drawn
# from Exp(1) and event rates λ_i from 10·Exp(1), adjusted so that
# λ_i > s_i (slow condition).
#
# Outputs:
#   • CSV file with per-p summary statistics.
#   • pgfplots coordinates for inclusion in the paper.
#
# Usage:
#   julia experiments/variance_timing_figure.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LinearAlgebra
using Random
using Printf
using Statistics

# ── Helpers ──────────────────────────────────────────────────────────────

"""
    stat_dist_Q(Q) -> RowVector

Stationary distribution of the CTMC with generator `Q`.
"""
function stat_dist_Q(Q)
    p = size(Q, 1)
    A = vcat(Q'[1:(p-1), :], ones(p)')
    b = vcat(zeros(p-1), 1.0)
    return (A \ b)'
end

"""
    match_mmtcp_to_smmpp(Q, λ) -> Matrix

Construct the 2p×2p generator of the MMTCP associated with an SMMPP
having background generator `Q` and event-rate vector `λ`.
"""
function match_mmtcp_to_smmpp(Q, λ)
    p = size(Q, 1)
    s = [sum(Q[i,j] for j in setdiff(1:p, i)) for i in 1:p]
    Q_m = zeros(2p, 2p)
    for i in 1:p, j in 1:p
        h, k = 2(i-1)+1, 2(j-1)+1
        if i == j
            Q_m[h, k]   = -λ[i];        Q_m[h, k+1]   = λ[i] - s[i]
            Q_m[h+1, k] = λ[i] - s[i];  Q_m[h+1, k+1] = -λ[i]
        else
            Q_m[h, k]     = Q[i,j]
            Q_m[h+1, k+1] = Q[i,j]
        end
    end
    Q_m
end

"""
    make_random_smmpp(p; rng) -> (Q, λ)

Generate a random SMMPP_p.  Off-diagonal entries of Q are drawn from Exp(1);
event rates λ_i from 10·Exp(1), adjusted so that λ_i > s_i (slow condition).
"""
function make_random_smmpp(p; rng=Random.default_rng())
    Q = zeros(p, p)
    for i in 1:p, j in 1:p
        i == j && continue
        Q[i, j] = randexp(rng)
    end
    for i in 1:p
        Q[i, i] = -sum(Q[i, j] for j in 1:p if j != i)
    end
    s = [-Q[i, i] for i in 1:p]

    λ = [10.0 * randexp(rng) for _ in 1:p]
    for i in 1:p
        if λ[i] <= s[i]
            λ[i] = s[i] + 5.0 * randexp(rng)
        end
    end
    return Q, λ
end

"""
    event_moment(k, π, C, λ★, 𝟏) -> Float64

k-th moment of the inter-event time for a MAP with parameters (C, D),
stationary distribution π, and overall rate λ★.
"""
function event_moment(k, π_vec, C, λ_star, one_vec)
    (-1)^(k+1) * factorial(k) / λ_star * (π_vec * inv(C)^(k-1) * one_vec)
end

"""
    bench(f; target_seconds, min_samples) -> Float64

Minimum wall time (nanoseconds) for calling `f()`, estimated via repeated
evaluation over at least `min_samples` rounds or `target_seconds`.
"""
function bench(f; target_seconds=0.3, min_samples=5)
    f()  # warmup
    t0 = time_ns(); f(); t_one = time_ns() - t0
    evals_per = max(1, ceil(Int, 1_000_000 / max(t_one, 1)))
    best = Inf
    deadline = time() + target_seconds
    n = 0
    while n < min_samples || (time() < deadline && n < 5000)
        t0 = time_ns()
        for _ in 1:evals_per; f(); end
        elapsed = (time_ns() - t0) / evals_per
        best = min(best, elapsed)
        n += 1
    end
    best
end

# ── Setup ────────────────────────────────────────────────────────────────

"""
    setup_maps(p; rng) -> NamedTuple

Return named tuples `(smmpp, mmtcp)` each containing the stationary
distribution π, matrix C, overall rate λ★, and ones vector needed for
moment computations.
"""
function setup_maps(p; rng=Random.default_rng())
    Q_s, λ = make_random_smmpp(p; rng=rng)
    D_s = Diagonal(float.(λ))
    C_s = Q_s - D_s
    one_p = ones(p)
    π_s = stat_dist_Q(Q_s)
    λ_star = (π_s * D_s * one_p)[1]

    Q_m = match_mmtcp_to_smmpp(Q_s, λ)
    D_m = Q_m - Diagonal(Q_m)
    C_m = Diagonal(diag(Q_m))
    one_2p = ones(2p)
    π_m = stat_dist_Q(Q_m)
    λ_star_m = (π_m * D_m * one_2p)[1]

    (smmpp = (π=π_s, C=C_s, λ_star=λ_star, one=one_p),
     mmtcp = (π=π_m, C=C_m, λ_star=λ_star_m, one=one_2p))
end

"""Inter-event standard deviation σ_T = √(m₂ − m₁²)."""
function inter_event_std(π_vec, C, λ_star, one_vec)
    m1 = event_moment(1, π_vec, C, λ_star, one_vec)
    m2 = event_moment(2, π_vec, C, λ_star, one_vec)
    sqrt(m2 - m1^2)
end

"""Relative difference |a − b| / max(|a|, |b|)."""
function reldiff(a, b)
    denom = max(abs(a), abs(b))
    denom < 1e-300 ? 0.0 : abs(a - b) / denom
end

# ── Main experiment ──────────────────────────────────────────────────────

function run_experiment(; p_values=5:5:100, n_instances=200, seed=42)
    rng = MersenneTwister(seed)
    results = []

    for p in p_values
        @printf("p = %3d ... ", p)
        t_start = time()

        times_s  = Float64[]
        times_m  = Float64[]
        rel_diffs = Float64[]

        for inst in 1:n_instances
            maps = setup_maps(p; rng=rng)
            s = maps.smmpp;  m = maps.mmtcp

            # Timing: collect a few samples per p (stable across instances)
            if inst <= 3
                t_s = bench(() -> inter_event_std(s.π, s.C, s.λ_star, s.one))
                t_m = bench(() -> inter_event_std(m.π, m.C, m.λ_star, m.one))
                push!(times_s, t_s)
                push!(times_m, t_m)
            end

            # Relative difference for every instance
            v_s = inter_event_std(s.π, s.C, s.λ_star, s.one)
            v_m = inter_event_std(m.π, m.C, m.λ_star, m.one)
            push!(rel_diffs, reldiff(v_s, v_m))
        end

        med_t_s  = median(times_s)
        med_t_m  = median(times_m)
        rd_med   = median(rel_diffs)
        rd_q05   = quantile(rel_diffs, 0.05)
        rd_q95   = quantile(rel_diffs, 0.95)

        push!(results, (p=p, t_s=med_t_s, t_m=med_t_m,
                        rd_median=rd_med, rd_q05=rd_q05, rd_q95=rd_q95))

        @printf("done (%.1fs)  speedup=%.0f×  rel_diff=%.2f%% [%.2f%%, %.2f%%]\n",
                time()-t_start, med_t_s/med_t_m,
                rd_med*100, rd_q05*100, rd_q95*100)
    end

    return results
end

# ── Output helpers ───────────────────────────────────────────────────────

function save_csv(results, dir=@__DIR__)
    path = joinpath(dir, "variance_timing_data.csv")
    open(path, "w") do f
        println(f, "p,time_smmpp_ns,time_mmtcp_ns,reldiff_median,reldiff_q05,reldiff_q95")
        for r in results
            @printf(f, "%d,%.2f,%.2f,%.6e,%.6e,%.6e\n",
                    r.p, r.t_s, r.t_m, r.rd_median, r.rd_q05, r.rd_q95)
        end
    end
    println("Saved: $path")
end

function print_pgfplots_coords(results)
    println("\n% ── Timing figure coordinates (seconds) ──")
    println("% SMMPP (dense C)")
    print("\\addplot coordinates {")
    for r in results
        @printf(" (%d, %.4e)", r.p, r.t_s * 1e-9)
    end
    println(" };")

    println("% MMTCP (diagonal C)")
    print("\\addplot coordinates {")
    for r in results
        @printf(" (%d, %.4e)", r.p, r.t_m * 1e-9)
    end
    println(" };")

    println("\n% ── Relative difference figure coordinates (%) ──")
    println("% Upper band (95th percentile)")
    print("\\addplot[name path=upper10, draw=none] coordinates {")
    for r in results
        @printf(" (%d, %.3f)", r.p, r.rd_q95 * 100)
    end
    println(" };")

    println("% Lower band (5th percentile)")
    print("\\addplot[name path=lower10, draw=none] coordinates {")
    for r in results
        @printf(" (%d, %.3f)", r.p, r.rd_q05 * 100)
    end
    println(" };")

    println("% Median")
    print("\\addplot coordinates {")
    for r in results
        @printf(" (%d, %.3f)", r.p, r.rd_median * 100)
    end
    println(" };")
end

# ── Run ──────────────────────────────────────────────────────────────────

results = run_experiment()
save_csv(results)
print_pgfplots_coords(results)
