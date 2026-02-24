using LinearAlgebra
using Random

# --- Basic MAP helpers ---

"""
    map_generator(C, D)

Return the CTMC generator Q = C + D for a MAP(C, D).
"""
map_generator(C, D) = C + D

"""
    stat_dist(Q)

Stationary distribution π of a CTMC generator Q, returned as a row vector.
Solves π Q = 0, π 1 = 1.
"""
function stat_dist(Q)
    p = size(Q, 1)
    A = vcat(Q'[1:(p-1), :], ones(p)')
    b = vcat(zeros(p-1), 1.0)
    return (A \ b)'
end

"""
    map_arrival_rate(C, D)

Overall stationary arrival rate λ* = π D 1 for a MAP(C, D).
"""
function map_arrival_rate(C, D)
    Q = map_generator(C, D)
    π = stat_dist(Q)
    return (π * D * ones(size(D, 1)))[1]
end

"""
    check_map(C, D)

Basic sanity checks on MAP matrices (C, D).
Returns a named tuple of checks.
"""
function check_map(C, D)
    p = size(C, 1)
    Q = map_generator(C, D)
    row_sums = Q * ones(p)
    generator_ok = all(abs.(row_sums) .< 1e-10)
    D_nonneg = all(D .>= -1e-15)
    C_offdiag_nonneg = all(C - Diagonal(C) .>= -1e-15)
    C_diag_neg = all(diag(C) .< 0)
    return (generator_ok=generator_ok, D_nonneg=D_nonneg,
            C_offdiag_nonneg=C_offdiag_nonneg, C_diag_neg=C_diag_neg)
end

# --- QBD blocks for MAP/M/1 ---

"""
    qbd_blocks_map_m_1(C, D, μ)

Return the QBD block matrices (A0, A1, A2, B0, B1) for a MAP(C,D)/M/1 queue
with service rate μ.

- A0 = D          (arrivals, level up)
- A1 = C - μI     (same level, for levels ≥ 1)
- A2 = μI         (service, level down)
- B0 = D          (arrivals from level 0)
- B1 = C          (same level, at level 0 — no service when empty)
"""
function qbd_blocks_map_m_1(C, D, μ)
    p = size(C, 1)
    A0 = D
    A2 = μ * I(p)
    A1 = C - μ * I(p)
    B0 = D
    B1 = C
    return (A0=A0, A1=A1, A2=A2, B0=B0, B1=B1)
end

# --- Matrix-geometric rate matrix R ---

"""
    rate_matrix_R(A0, A1, A2; maxiter=10000, tol=1e-12)

Compute the rate matrix R for a QBD process by successive substitution.
R satisfies: A0 + R A1 + R² A2 = 0.

Iterates:  R_{n+1} = -A0 * inv(A1 + R_n * A2)
"""
function rate_matrix_R(A0, A1, A2; maxiter=10000, tol=1e-12, verbose=true)
    p = size(A0, 1)
    R = zeros(p, p)
    for iter in 1:maxiter
        R_new = -A0 * inv(A1 + R * A2)
        if maximum(abs.(R_new - R)) < tol
            return R_new
        end
        R = R_new
    end
    verbose && @warn "rate_matrix_R did not converge in $maxiter iterations"
    return R
end

"""
    rate_matrix_R(C, D, μ; kwargs...)

Convenience: compute R directly from MAP(C,D)/M/1 parameters.
"""
function rate_matrix_R(C, D, μ::Real; kwargs...)
    blk = qbd_blocks_map_m_1(C, D, μ)
    return rate_matrix_R(blk.A0, blk.A1, blk.A2; kwargs...)
end

# --- Stationary distribution of the QBD ---

"""
    qbd_pi0(R, A2, B1)

Compute π_0 (the level-0 stationary probability row vector) from:
  π_0 (B1 + R A2) = 0,  π_0 (I - R)^{-1} 1 = 1.
Level-0 balance: π_0 B1 + π_1 A2 = 0, with π_1 = π_0 R.
"""
function qbd_pi0(R, A2, B1)
    p = size(R, 1)
    M = B1 + R * A2

    inv_I_minus_R = inv(I(p) - R)
    norm_vec = inv_I_minus_R * ones(p)

    A = vcat(M'[1:(p-1), :], norm_vec')
    b = vcat(zeros(p-1), 1.0)
    π0 = (A \ b)'
    return π0
end

"""
    qbd_stationary_map_m_1(C, D, μ; R_kwargs...)

Full stationary distribution computation for MAP(C,D)/M/1.
Returns (π0, R, ρ) where:
  - π0: level-0 stationary row vector
  - R: rate matrix
  - ρ: traffic intensity λ*/μ
"""
function qbd_stationary_map_m_1(C, D, μ; kwargs...)
    λ_star = map_arrival_rate(C, D)
    ρ = λ_star / μ
    if ρ >= 1.0
        @warn "System is unstable: ρ = $ρ ≥ 1"
    end

    blk = qbd_blocks_map_m_1(C, D, μ)
    R = rate_matrix_R(blk.A0, blk.A1, blk.A2; kwargs...)
    π0 = qbd_pi0(R, blk.A2, blk.B1)
    return (π0=π0, R=R, ρ=ρ)
end

# --- Performance measures ---

"""
    level_prob(π0, R, k)

Stationary probability row vector at level k: π_k = π_0 R^k.
"""
level_prob(π0, R, k) = π0 * R^k

"""
    mean_queue_length(π0, R)

Mean number in system: E[N] = π_0 R (I - R)^{-2} 1.
"""
function mean_queue_length(π0, R)
    p = size(R, 1)
    inv_I_minus_R = inv(I(p) - R)
    return (π0 * R * inv_I_minus_R^2 * ones(p))[1]
end

"""
    second_factorial_moment_queue_length(π0, R)

Second factorial moment E[N(N-1)] = 2 π_0 R^2 (I - R)^{-3} 1.
"""
function second_factorial_moment_queue_length(π0, R)
    p = size(R, 1)
    inv_I_minus_R = inv(I(p) - R)
    return (2 * π0 * R^2 * inv_I_minus_R^3 * ones(p))[1]
end

"""
    var_queue_length(π0, R)

Variance of the queue length: Var(N) = E[N(N-1)] + E[N] - E[N]^2.
"""
function var_queue_length(π0, R)
    EN = mean_queue_length(π0, R)
    EN2_fact = second_factorial_moment_queue_length(π0, R)
    return EN2_fact + EN - EN^2
end

"""
    std_queue_length(π0, R)

Standard deviation of the queue length.
"""
std_queue_length(π0, R) = sqrt(var_queue_length(π0, R))

"""
    prob_empty(π0)

Probability system is empty: P(N=0) = π_0 * 1.
"""
prob_empty(π0) = (π0 * ones(size(π0, 2)))[1]

"""
    prob_level_k(π0, R, k)

Total probability of being at level k.
"""
prob_level_k(π0, R, k) = (level_prob(π0, R, k) * ones(size(R, 1)))[1]

# --- Gillespie Monte-Carlo simulation ---

"""
    simulate_map_m_1(C, D, μ; T_max=10^5, max_levels=1000, seed=nothing)

Gillespie simulation of a MAP(C,D)/M/1 queue. Accumulates sojourn time
in each (level, phase) state — constant memory usage (no trace stored).

Returns a named tuple:
  - mean_queue_length: time-average E[N]
  - prob_empty: fraction of time system is empty
  - level_probs: vector of P(N=k) for k = 0, ..., max observed level
  - T_max: total simulated time
"""
function simulate_map_m_1(C, D, μ; T_max=100000.0, max_levels=1000, seed=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end

    p = size(C, 1)
    Q = map_generator(C, D)

    # Sojourn time accumulator: rows = levels 0..max_levels, cols = phases 1..p
    sojourn = zeros(max_levels + 1, p)
    max_level_seen = 0

    # Initial state: empty queue, phase drawn from stationary dist of Q
    n = 0   # queue length
    phase = 1

    t = 0.0
    while t < T_max
        # Total rate out of current state
        # Possible events:
        #   1) MAP hidden transition (C off-diagonal): phase changes, n unchanged
        #   2) MAP arrival (D row):                    phase may change, n -> n+1
        #   3) Service completion (rate μ if n ≥ 1):   n -> n-1, phase unchanged
        rates = Float64[]
        events = Tuple{Int,Int}[]  # (new_n, new_phase)

        # C off-diagonal transitions (phase change, no arrival)
        for j in 1:p
            if j != phase
                r = C[phase, j]
                if r > 0
                    push!(rates, r)
                    push!(events, (n, j))
                end
            end
        end

        # D transitions (arrival)
        for j in 1:p
            r = D[phase, j]
            if r > 0
                push!(rates, r)
                push!(events, (n + 1, j))
            end
        end

        # Service
        if n > 0
            push!(rates, μ)
            push!(events, (n - 1, phase))
        end

        total_rate = sum(rates)

        # Time to next event
        dt = -log(rand()) / total_rate
        if t + dt > T_max
            dt = T_max - t
        end

        # Accumulate sojourn time
        level_idx = min(n, max_levels) + 1
        sojourn[level_idx, phase] += dt
        max_level_seen = max(max_level_seen, n)

        t += dt
        if t >= T_max
            break
        end

        # Choose event
        u = rand() * total_rate
        cumsum_r = 0.0
        chosen = length(events)
        for i in eachindex(rates)
            cumsum_r += rates[i]
            if u <= cumsum_r
                chosen = i
                break
            end
        end

        n, phase = events[chosen]
    end

    # Compute results
    K = min(max_level_seen, max_levels)
    level_probs = vec(sum(sojourn[1:(K+1), :], dims=2)) ./ T_max
    mean_n = sum(k * level_probs[k+1] for k in 0:K)
    mean_n2 = sum(k^2 * level_probs[k+1] for k in 0:K)
    var_n = mean_n2 - mean_n^2
    std_n = sqrt(max(var_n, 0.0))
    p_empty = level_probs[1]

    return (mean_queue_length=mean_n, var_queue_length=var_n, std_queue_length=std_n,
            prob_empty=p_empty, level_probs=level_probs, T_max=T_max)
end
