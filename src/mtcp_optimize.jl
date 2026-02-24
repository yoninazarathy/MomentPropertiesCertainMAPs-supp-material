using Optim

# Assumes map_m_1_helpers.jl is already included
# (provides mean_queue_length, std_queue_length, etc.)

# --- MTCP helpers ---

"""
    mtcp_from_generator(Q)

Decompose a CTMC generator Q into MTCP form (C, D) where:
  C = diag(Q)  (diagonal matrix of negative departure rates)
  D = Q - diag(Q)  (off-diagonal: every transition is an arrival)
"""
function mtcp_from_generator(Q)
    C = Diagonal(diag(Q))
    D = Q - C
    return (C=Matrix(C), D=D)
end

"""
    generator_from_offdiag(x, q)

Build a q×q CTMC generator from a vector x of q(q-1) non-negative off-diagonal
entries (row-major order). Diagonal is set so rows sum to zero.
"""
function generator_from_offdiag(x, q)
    Q = zeros(q, q)
    idx = 1
    for i in 1:q
        for j in 1:q
            if i != j
                Q[i, j] = x[idx]
                idx += 1
            end
        end
    end
    for i in 1:q
        Q[i, i] = -sum(Q[i, j] for j in 1:q if j != i)
    end
    return Q
end

# --- MMTCP construction ---

"""
    match_mmtcp_to_smmpp(Q, λ)

Construct a 2p-phase MMTCP generator from a p-phase SMMPP with generator Q
and arrival rates λ. Each SMMPP phase i expands to a 2×2 diagonal block Λ(i),
and each off-diagonal block (i,j) becomes Q[i,j]·I₂.
"""
function match_mmtcp_to_smmpp(Q, λ)
    p = size(Q)[1]
    s = [sum(Q[i,j] for j in setdiff(1:p,i)) for i in 1:p]
    Λ(i) = [-λ[i]       λ[i]-s[i];
            λ[i]-s[i]  -λ[i]]
    H(i,j) = Diagonal(fill(Q[i,j], 2))
    Q_mmtcp = zeros(2*p, 2*p)
    for i in 1:p
        for j in 1:p
            h = 2*(i-1)+1
            k = 2*(j-1)+1
            if i == j
                Q_mmtcp[h:h+1,k:k+1] = Λ(i)
            else
                Q_mmtcp[h:h+1,k:k+1] = H(i,j)
            end
        end
    end
    Q_mmtcp
end

"""
    offdiag_vector(Q)

Extract off-diagonal entries of Q in row-major order (matching generator_from_offdiag).
"""
function offdiag_vector(Q)
    q = size(Q, 1)
    x = Float64[]
    for i in 1:q
        for j in 1:q
            if i != j
                push!(x, Q[i, j])
            end
        end
    end
    return x
end

# --- Scaling helper ---

"""
    scale_generator_to_rate(Q, λ_target)

Return αQ where α is chosen so that the MTCP with generator αQ has
arrival rate exactly λ_target. Since π is invariant to uniform scaling
of Q, we have λ*(αQ) = α λ*(Q), so α = λ_target / λ*(Q).
"""
function scale_generator_to_rate(Q, λ_target)
    C, D = mtcp_from_generator(Q)
    λ_star = map_arrival_rate(C, D)
    α = λ_target / λ_star
    return α * Q
end

# --- Optimization ---

"""
    optimize_mtcp(C_target, D_target, μ; q, Q_init, restarts, iterations, seed)

Find a q-phase MTCP whose MAP(C,D)/M/1 queue length distribution best matches
the target MAP(C_target, D_target)/M/1 queue, with the same service rate μ.

The arrival rate λ* is matched exactly by construction (generator scaling).
Two-phase optimization:
  Phase 1: minimize E[N] error to find a feasible starting point.
  Phase 2: minimize Std[N] error with E[N] penalty, warm-started from phase 1.

If `Q_init` is provided (e.g. an MMTCP generator), restart 1 is warm-started
from it and the remaining restarts use random initial points.
"""
function optimize_mtcp(C_target, D_target, μ;
                       q = size(C_target, 1),
                       Q_init = nothing,
                       penalty = 1e8,
                       restarts = 10,
                       iterations = 50000,
                       seed = nothing)
    if seed !== nothing
        Random.seed!(seed)
    end

    # Compute target quantities
    res_target = qbd_stationary_map_m_1(C_target, D_target, μ)
    λ_target = map_arrival_rate(C_target, D_target)
    EN_target = mean_queue_length(res_target.π0, res_target.R)
    StdN_target = std_queue_length(res_target.π0, res_target.R)

    n_params = q * (q - 1)

    # Helper: build scaled MTCP and solve QBD, returns nothing on failure
    function solve_mtcp(θ)
        x = exp.(θ)
        Q_raw = generator_from_offdiag(x, q)
        Q = scale_generator_to_rate(Q_raw, λ_target)
        C, D = mtcp_from_generator(Q)
        blk = qbd_blocks_map_m_1(C, D, μ)
        R = rate_matrix_R(blk.A0, blk.A1, blk.A2; verbose=false)
        π0 = qbd_pi0(R, blk.A2, blk.B1)
        return (π0=π0, R=R, C=C, D=D, Q=Q)
    end

    # Phase 1 objective: just match E[N]
    function obj_phase1(θ)
        try
            sol = solve_mtcp(θ)
            EN = mean_queue_length(sol.π0, sol.R)
            return ((EN - EN_target) / EN_target)^2
        catch
            return 1e6
        end
    end

    # Phase 2 objective: match Std[N] with E[N] penalty
    function obj_phase2(θ)
        try
            sol = solve_mtcp(θ)
            EN = mean_queue_length(sol.π0, sol.R)
            StdN = std_queue_length(sol.π0, sol.R)
            EN_err = ((EN - EN_target) / EN_target)^2
            StdN_err = ((StdN - StdN_target) / StdN_target)^2
            return StdN_err + penalty * EN_err
        catch
            return 1e6
        end
    end

    # Convert Q_init to θ-space if provided
    θ_init = nothing
    if Q_init !== nothing
        @assert size(Q_init, 1) == q "Q_init has $(size(Q_init,1)) phases but q=$q"
        x_init = offdiag_vector(Q_init)
        # Replace zeros with small epsilon before taking log
        ε = 1e-6
        x_init = [max(xi, ε) for xi in x_init]
        θ_init = log.(x_init)
    end

    # Multi-start
    best_val = Inf
    best_θ = nothing
    println("Target: λ*=$(round(λ_target, digits=3)), E[N]=$(round(EN_target, digits=3)), Std[N]=$(round(StdN_target, digits=3)), q=$q")

    for r in 1:restarts
        if r == 1 && θ_init !== nothing
            θ0 = θ_init
            label = "  restart $r/$restarts [MMTCP init]:"
        else
            θ0 = 0.5 * randn(n_params)
            label = "  restart $r/$restarts:"
        end
        try
            # Phase 1: find E[N]-feasible point
            res1 = optimize(obj_phase1, θ0, NelderMead(),
                            Optim.Options(iterations=iterations))
            θ_warm = Optim.minimizer(res1)
            EN1 = try
                sol = solve_mtcp(θ_warm)
                mean_queue_length(sol.π0, sol.R)
            catch; NaN end
            print("$label phase1 E[N]=$(round(EN1, digits=3))")

            # Phase 2: refine for Std[N], warm-started
            res2 = optimize(obj_phase2, θ_warm, NelderMead(),
                            Optim.Options(iterations=iterations))

            val = Optim.minimum(res2)
            sol2 = try solve_mtcp(Optim.minimizer(res2)) catch; nothing end
            EN2 = sol2 !== nothing ? mean_queue_length(sol2.π0, sol2.R) : NaN
            StdN2 = sol2 !== nothing ? std_queue_length(sol2.π0, sol2.R) : NaN
            star = val < best_val ? " *best*" : ""
            println(" → phase2 Std[N]=$(round(StdN2, digits=5)) E[N]=$(round(EN2, digits=3))$star")

            if val < best_val
                best_val = val
                best_θ = Optim.minimizer(res2)
            end
        catch e
            println("$label failed ($e)")
            continue
        end
    end

    if best_θ === nothing
        error("All optimization restarts failed")
    end

    # Extract best solution
    sol = solve_mtcp(best_θ)
    EN = mean_queue_length(sol.π0, sol.R)
    StdN = std_queue_length(sol.π0, sol.R)

    return (Q_mtcp=sol.Q, C_mtcp=sol.C, D_mtcp=sol.D,
            result_mtcp=(π0=sol.π0, R=sol.R, ρ=λ_target/μ),
            result_target=res_target,
            EN=EN, EN_target=EN_target,
            StdN=StdN, StdN_target=StdN_target)
end

# --- Gradient-based optimization ---

"""
    optimize_mtcp_lbfgs(C_target, D_target, μ; q, Q_init, restarts, iterations, seed)

Same goal as `optimize_mtcp` but uses LBFGS (gradient-based, finite differences).

Combined objective: weighted sum of relative E[N] and Std[N] errors.
No two-phase split — gradient methods handle the combined objective well.

When `Q_init` is provided, restart 1 starts directly from the MMTCP (skipping
Phase 1, since it already has good E[N]). Random restarts still use Phase 1
to get a feasible starting point before the combined objective.
"""
function optimize_mtcp_lbfgs(C_target, D_target, μ;
                             q = size(C_target, 1),
                             Q_init = nothing,
                             penalty = 1e8,
                             restarts = 10,
                             iterations = 200,
                             phase1_iterations = 500,
                             show_every = 50,
                             seed = nothing)
    if seed !== nothing
        Random.seed!(seed)
    end

    # Compute target quantities
    res_target = qbd_stationary_map_m_1(C_target, D_target, μ)
    λ_target = map_arrival_rate(C_target, D_target)
    EN_target = mean_queue_length(res_target.π0, res_target.R)
    StdN_target = std_queue_length(res_target.π0, res_target.R)

    n_params = q * (q - 1)

    # Helper: build scaled MTCP and solve QBD
    function solve_mtcp(θ)
        x = exp.(θ)
        Q_raw = generator_from_offdiag(x, q)
        Q = scale_generator_to_rate(Q_raw, λ_target)
        C, D = mtcp_from_generator(Q)
        blk = qbd_blocks_map_m_1(C, D, μ)
        R = rate_matrix_R(blk.A0, blk.A1, blk.A2; verbose=false)
        π0 = qbd_pi0(R, blk.A2, blk.B1)
        return (π0=π0, R=R, C=C, D=D, Q=Q)
    end

    # Phase 1 objective (for random starts): just match E[N] with Nelder-Mead
    function obj_phase1(θ)
        try
            sol = solve_mtcp(θ)
            EN = mean_queue_length(sol.π0, sol.R)
            return ((EN - EN_target) / EN_target)^2
        catch
            return 1e6
        end
    end

    # Combined objective for LBFGS: Std[N] error + heavy E[N] penalty
    function obj_combined(θ)
        try
            sol = solve_mtcp(θ)
            EN = mean_queue_length(sol.π0, sol.R)
            StdN = std_queue_length(sol.π0, sol.R)
            EN_err = ((EN - EN_target) / EN_target)^2
            StdN_err = ((StdN - StdN_target) / StdN_target)^2
            return StdN_err + penalty * EN_err
        catch
            return 1e6
        end
    end

    # Convert Q_init to θ-space if provided
    θ_init = nothing
    if Q_init !== nothing
        @assert size(Q_init, 1) == q "Q_init has $(size(Q_init,1)) phases but q=$q"
        x_init = offdiag_vector(Q_init)
        # Replace zeros with a fraction of the mean nonzero entry to avoid
        # extreme log values that cause LBFGS scaling issues
        nonzero_vals = filter(v -> v > 0, x_init)
        ε = isempty(nonzero_vals) ? 0.1 : sum(nonzero_vals) / length(nonzero_vals) * 0.01
        x_init = [max(xi, ε) for xi in x_init]
        θ_init = log.(x_init)
    end

    best_val = Inf
    best_θ = nothing
    restart_stats = []
    println("Target: λ*=$(round(λ_target, digits=3)), E[N]=$(round(EN_target, digits=3)), Std[N]=$(round(StdN_target, digits=3)), q=$q [LBFGS]")

    for r in 1:restarts
        use_mmtcp = (r == 1 && θ_init !== nothing)
        label = use_mmtcp ? "  restart $r/$restarts [MMTCP init]:" : "  restart $r/$restarts:"

        try
            t_start = time()
            phase1_f_calls = 0
            phase1_iters = 0

            if use_mmtcp
                # Skip Phase 1: MMTCP already has good E[N], go straight to LBFGS
                θ_start = θ_init
                println("$label skip phase1 → LBFGS")
            else
                # Phase 1: Nelder-Mead to get E[N]-feasible point
                print("$label phase1 (NelderMead)...")
                θ0 = 0.5 * randn(n_params)
                res1 = optimize(obj_phase1, θ0, NelderMead(),
                                Optim.Options(iterations=phase1_iterations))
                θ_start = Optim.minimizer(res1)
                phase1_f_calls = Optim.f_calls(res1)
                phase1_iters = Optim.iterations(res1)
                EN1 = try
                    sol = solve_mtcp(θ_start)
                    mean_queue_length(sol.π0, sol.R)
                catch; NaN end
                println(" done ($(phase1_iters) iters, $(phase1_f_calls) f_calls), E[N]=$(round(EN1, digits=3)) → LBFGS")
            end

            # LBFGS refinement with trace
            res2 = optimize(obj_combined, θ_start, LBFGS(),
                            Optim.Options(iterations=iterations, store_trace=true,
                                          show_trace=true, show_every=show_every))
            elapsed = time() - t_start

            val = Optim.minimum(res2)
            sol2 = try solve_mtcp(Optim.minimizer(res2)) catch; nothing end
            EN2 = sol2 !== nothing ? mean_queue_length(sol2.π0, sol2.R) : NaN
            StdN2 = sol2 !== nothing ? std_queue_length(sol2.π0, sol2.R) : NaN
            star = val < best_val ? " *best*" : ""
            println("\r$label DONE Std[N]=$(round(StdN2, digits=5)) E[N]=$(round(EN2, digits=3)) [$(Optim.iterations(res2)) iters, $(Optim.f_calls(res2)) f_calls, $(round(elapsed, digits=1))s]$star                    ")

            # Collect trace: objective value at each LBFGS iteration
            lbfgs_trace = Optim.f_trace(res2)

            push!(restart_stats, (
                restart = r,
                is_mmtcp = use_mmtcp,
                phase1_f_calls = phase1_f_calls,
                phase1_iters = phase1_iters,
                lbfgs_iters = Optim.iterations(res2),
                lbfgs_f_calls = Optim.f_calls(res2),
                total_f_calls = phase1_f_calls + Optim.f_calls(res2),
                elapsed = elapsed,
                final_val = val,
                EN = EN2,
                StdN = StdN2,
                obj_trace = lbfgs_trace
            ))

            if val < best_val
                best_val = val
                best_θ = Optim.minimizer(res2)
            end
        catch e
            println("$label failed ($e)")
            continue
        end
    end

    if best_θ === nothing
        error("All optimization restarts failed")
    end

    sol = solve_mtcp(best_θ)
    EN = mean_queue_length(sol.π0, sol.R)
    StdN = std_queue_length(sol.π0, sol.R)

    return (Q_mtcp=sol.Q, C_mtcp=sol.C, D_mtcp=sol.D,
            result_mtcp=(π0=sol.π0, R=sol.R, ρ=λ_target/μ),
            result_target=res_target,
            EN=EN, EN_target=EN_target,
            StdN=StdN, StdN_target=StdN_target,
            restart_stats=restart_stats)
end
