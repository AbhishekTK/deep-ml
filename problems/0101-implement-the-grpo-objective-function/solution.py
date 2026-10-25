import numpy as np

def grpo_objective(rhos, A, pi_theta_old, pi_theta_ref, epsilon=0.2, beta=0.01) -> float:
    """
    Compute the GRPO objective function.

    Args:
        rhos: List of likelihood ratios (pi_theta / pi_theta_old).
        A: List of advantage estimates.
        pi_theta_old: List of old policy probabilities (per-sample, not normalized).
        pi_theta_ref: List of reference policy probabilities (per-sample, not normalized).
        epsilon: Clipping parameter for the surrogate objective.
        beta: KL divergence penalty coefficient.

    Returns:
        The computed GRPO objective value.
    """
    # Your code here
    # pass
    rhos = np.asarray(rhos)
    A = np.asarray(A)
    pi_theta_old = np.asarray(pi_theta_old)
    pi_theta_ref = np.asarray(pi_theta_ref)

    pi_theta = rhos*pi_theta_old

    surrogate_unclipped = rhos*A
    surrogate_clipped = np.clip(rhos,1.0-epsilon,1.0+epsilon)*A
    policy_objective = np.minimum(surrogate_unclipped, surrogate_clipped)

    # 3. Compute per-token/per-sample KL divergence: D_KL(pi_theta || pi_ref)
    # KL = pi_theta * log(pi_theta/pi_ref)
    ratio = pi_theta_ref/pi_theta
    kl_divergence = rhos * (ratio - np.log(ratio)-1.0)

    total_objective = policy_objective - beta*kl_divergence
    return float(np.mean(total_objective))








        