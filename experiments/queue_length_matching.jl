# experiments/queue_length_matching.jl
#
# MAP/M/1 queue-length matching: MMTCP warm start vs random start.
#
# For each random slow SMMPP_p with service rate μ (traffic intensity
# ρ ∈ [0.5, 0.9]), we seek a 2p-phase MTCP whose MAP/M/1 queue has the
# same E[N] and Std[N].  Two optimisation strategies are compared:
#
#   (a) MMTCP start  — Phase 2 only (Nelder–Mead on the combined
#       Std[N] + penalty·E[N] objective), initialised from the matched
#       MMTCP_{2p} generator of Theorem 4.3.
#
#   (b) Random start — Phase 1 (E[N] matching) then Phase 2, both
#       with the same per-phase Nelder–Mead budget.
#
# The random start therefore receives strictly more total computation
# (two phases vs one), so any advantage of the MMTCP start is due
# entirely to the quality of the initial point.
#
# Output:  per-trial log to stdout;  summary table;  CSV file.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LinearAlgebra
using Random
using Optim
using Printf
using Statistics

include(joinpath(@__DIR__, "..", "src", "map_m_1_helpers.jl"))
include(joinpath(@__DIR__, "..", "src", "mtcp_optimize.jl"))

# ── Random SMMPP generation ────────────────────────────────────

"""
    random_smmpp(p; rng)

Generate a random p-phase slow SMMPP: off-diagonal entries of Q drawn
from Exp(1), arrival rates λ_i drawn from 10·Exp(1) and adjusted so
that λ_i > s_i (slow condition) by setting λ_i = s_i + 5·Exp(1) when
needed.  Traffic intensity ρ ~ Uniform(0.5, 0.9) determines μ.
"""
function random_smmpp(p; rng=Random.default_rng())
    Q = zeros(p, p)
    for i in 1:p, j in 1:p
        i == j && continue
        Q[i, j] = randexp(rng)
    end
    for i in 1:p
        Q[i, i] = -sum(Q[i, j] for j in 1:p if j != i)
    end
    s = [-Q[i, i] for i in 1:p]

    λ_vec = [10.0 * randexp(rng) for _ in 1:p]
    for i in 1:p
        if λ_vec[i] <= s[i]
            λ_vec[i] = s[i] + 5.0 * randexp(rng)
        end
    end

    C = Q - Diagonal(λ_vec)
    D = Diagonal(λ_vec)
    λ_star = map_arrival_rate(C, D)
    ρ = 0.5 + 0.4 * rand(rng)
    μ = λ_star / ρ

    return (Q=Q, λ_vec=λ_vec, C=C, D=D, μ=μ, ρ=ρ)
end

# ── Single trial ────────────────────────────────────────────────

function run_trial(Q_smmpp, λ_vec, C, D, μ;
                   iterations=500, penalty=1e8)
    p = size(Q_smmpp, 1)
    q = 2p
    n_params = q * (q - 1)

    # Target queue-length statistics
    res_t = qbd_stationary_map_m_1(C, D, μ)
    λ_t   = map_arrival_rate(C, D)
    EN_t  = mean_queue_length(res_t.π0, res_t.R)
    StdN_t = std_queue_length(res_t.π0, res_t.R)

    # Build and solve MTCP from parameter vector θ
    function solve_mtcp(θ)
        x = exp.(θ)
        Q_raw = generator_from_offdiag(x, q)
        Q_sc  = scale_generator_to_rate(Q_raw, λ_t)
        Cm, Dm = mtcp_from_generator(Q_sc)
        blk = qbd_blocks_map_m_1(Cm, Dm, μ)
        R  = rate_matrix_R(blk.A0, blk.A1, blk.A2; verbose=false, maxiter=500)
        π0 = qbd_pi0(R, blk.A2, blk.B1)
        return (π0=π0, R=R)
    end

    obj_mean(θ) = try
        s = solve_mtcp(θ)
        ((mean_queue_length(s.π0, s.R) - EN_t) / EN_t)^2
    catch; 1e6 end

    obj_both(θ) = try
        s = solve_mtcp(θ)
        EN  = mean_queue_length(s.π0, s.R)
        StdN = std_queue_length(s.π0, s.R)
        ((StdN - StdN_t) / StdN_t)^2 + penalty * ((EN - EN_t) / EN_t)^2
    catch; 1e6 end

    # ── Raw MMTCP (before optimisation) ──
    Q_mmtcp = match_mmtcp_to_smmpp(Q_smmpp, λ_vec)
    Cm_raw, Dm_raw = mtcp_from_generator(Q_mmtcp)
    res_raw = qbd_stationary_map_m_1(Cm_raw, Dm_raw, μ)
    EN_raw  = mean_queue_length(res_raw.π0, res_raw.R)
    StdN_raw = std_queue_length(res_raw.π0, res_raw.R)

    # ── (a) MMTCP start: Phase 2 only ──
    x_init  = offdiag_vector(Q_mmtcp)
    nonzero = filter(>(0), x_init)
    ε = isempty(nonzero) ? 0.1 : mean(nonzero) * 0.01
    θ_mmtcp = log.([max(xi, ε) for xi in x_init])

    t0 = time()
    res_m = optimize(obj_both, θ_mmtcp, NelderMead(),
                     Optim.Options(iterations=iterations))
    wall_m = time() - t0

    sol_m  = solve_mtcp(Optim.minimizer(res_m))
    EN_m   = mean_queue_length(sol_m.π0, sol_m.R)
    StdN_m = std_queue_length(sol_m.π0, sol_m.R)

    # ── (b) Random start: Phase 1 + Phase 2 ──
    θ0 = 0.5 * randn(n_params)

    t0 = time()
    res1 = optimize(obj_mean, θ0, NelderMead(),
                    Optim.Options(iterations=iterations))
    res2 = optimize(obj_both, Optim.minimizer(res1), NelderMead(),
                    Optim.Options(iterations=iterations))
    wall_r = time() - t0

    sol_r  = solve_mtcp(Optim.minimizer(res2))
    EN_r   = mean_queue_length(sol_r.π0, sol_r.R)
    StdN_r = std_queue_length(sol_r.π0, sol_r.R)

    return (
        EN_t   = EN_t,   StdN_t  = StdN_t, ρ = λ_t / μ,
        # Raw MMTCP (no optimisation)
        EN_raw = EN_raw, StdN_raw = StdN_raw,
        EN_err_raw  = abs(EN_raw  - EN_t)  / EN_t,
        StdN_err_raw = abs(StdN_raw - StdN_t) / StdN_t,
        # MMTCP start (after optimisation)
        EN_m   = EN_m,   StdN_m  = StdN_m,
        EN_err_m  = abs(EN_m  - EN_t)  / EN_t,
        StdN_err_m = abs(StdN_m - StdN_t) / StdN_t,
        fcalls_m = Optim.f_calls(res_m),
        wall_m = wall_m,
        # Random start
        EN_r   = EN_r,   StdN_r  = StdN_r,
        EN_err_r  = abs(EN_r  - EN_t)  / EN_t,
        StdN_err_r = abs(StdN_r - StdN_t) / StdN_t,
        fcalls_r = Optim.f_calls(res1) + Optim.f_calls(res2),
        wall_r = wall_r,
    )
end

# ── Full experiment ─────────────────────────────────────────────

function run_experiment(; K=1000, p_values=[2, 3, 4, 5, 6],
                          iterations=500, penalty=1e8, seed=42)
    rng = MersenneTwister(seed)
    all_results = Dict{Int, Vector}()

    for p in p_values
        println("=" ^ 70)
        @printf("p = %d  (q = %d, %d free parameters, K = %d trials)\n",
                p, 2p, 2p*(2p-1), K)
        println("=" ^ 70)

        trials = []
        for k in 1:K
            smmpp = random_smmpp(p; rng=rng)
            trial = try
                run_trial(smmpp.Q, smmpp.λ_vec, smmpp.C, smmpp.D, smmpp.μ;
                          iterations=iterations, penalty=penalty)
            catch e
                @printf("Trial %3d: FAILED (%s)\n", k, e)
                continue
            end

            winner = trial.StdN_err_m < trial.StdN_err_r ? "MMTCP" : "Random"
            @printf("Trial %3d: ρ=%.2f  Raw=%.4f  MMTCP=%.4f  Random=%.4f  ← %s\n",
                    k, trial.ρ, trial.StdN_err_raw, trial.StdN_err_m, trial.StdN_err_r, winner)
            flush(stdout)
            push!(trials, trial)
        end

        all_results[p] = trials

        # Summary for this p
        n = length(trials)
        mmtcp_wins = count(t -> t.StdN_err_m < t.StdN_err_r, trials)
        println("\n", "-" ^ 70)
        @printf("p=%d summary (%d/%d valid trials):\n", p, n, K)
        @printf("  MMTCP wins:     %d/%d (%.0f%%)\n", mmtcp_wins, n, 100*mmtcp_wins/n)
        @printf("  Raw MMTCP (no opt):\n")
        @printf("    Mean  E[N] rel error:   %.4f\n", mean(t.EN_err_raw for t in trials))
        @printf("    Mean  Std[N] rel error: %.4f\n", mean(t.StdN_err_raw for t in trials))
        @printf("  After optimisation:\n")
        @printf("    Mean  Std[N] rel error — MMTCP: %.4f,  Random: %.4f\n",
                mean(t.StdN_err_m for t in trials),
                mean(t.StdN_err_r for t in trials))
        @printf("    Median Std[N] rel error — MMTCP: %.4f,  Random: %.4f\n",
                median([t.StdN_err_m for t in trials]),
                median([t.StdN_err_r for t in trials]))
        @printf("    Mean  E[N] rel error   — MMTCP: %.6f,  Random: %.6f\n",
                mean(t.EN_err_m for t in trials),
                mean(t.EN_err_r for t in trials))
        println("-" ^ 70, "\n")
    end

    return all_results
end

# ── Output ──────────────────────────────────────────────────────

function print_table(results; p_values=[2, 3, 4, 5, 6])
    println("\n", "=" ^ 95)
    println("Summary table for paper")
    println("=" ^ 95)
    @printf("  %-3s  %-5s  %-11s  %-14s  %-14s  %-14s  %-8s\n",
            "p", "q=2p", "MMTCP wins", "Raw MMTCP err", "Opt MMTCP err", "Random err", "Ratio")
    println("  ", "-" ^ 90)
    for p in p_values
        haskey(results, p) || continue
        trials = results[p]
        n = length(trials)
        wins = count(t -> t.StdN_err_m < t.StdN_err_r, trials)
        avg_raw = mean(t.StdN_err_raw for t in trials)
        avg_m   = mean(t.StdN_err_m for t in trials)
        avg_r   = mean(t.StdN_err_r for t in trials)
        @printf("  %-3d  %-5d  %4d/%-4d    %.4f          %.4f          %.4f        %.1f×\n",
                p, 2p, wins, n, avg_raw, avg_m, avg_r, avg_r/avg_m)
    end
    println("=" ^ 95)
end

function save_csv(results, path=joinpath(@__DIR__, "queue_length_matching.csv"))
    open(path, "w") do f
        println(f, "p,q,trial,rho,EN_target,StdN_target," *
                   "EN_err_raw,StdN_err_raw," *
                   "EN_err_mmtcp,StdN_err_mmtcp,fcalls_mmtcp,wall_mmtcp," *
                   "EN_err_random,StdN_err_random,fcalls_random,wall_random")
        for p in sort(collect(keys(results)))
            for (k, t) in enumerate(results[p])
                @printf(f, "%d,%d,%d,%.4f,%.6f,%.6f,%.6e,%.6e,%.6e,%.6e,%d,%.3f,%.6e,%.6e,%d,%.3f\n",
                        p, 2p, k, t.ρ, t.EN_t, t.StdN_t,
                        t.EN_err_raw, t.StdN_err_raw,
                        t.EN_err_m, t.StdN_err_m, t.fcalls_m, t.wall_m,
                        t.EN_err_r, t.StdN_err_r, t.fcalls_r, t.wall_r)
            end
        end
    end
    println("Saved: $path")
end

# ── Run ─────────────────────────────────────────────────────────

results = run_experiment(K=200, p_values=[2, 3, 4, 5])
print_table(results; p_values=[2, 3, 4, 5])
save_csv(results)
