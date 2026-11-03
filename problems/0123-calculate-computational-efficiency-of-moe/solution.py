def compute_efficiency(n_experts, k_active, d_in, d_out):
    """
    Calculate computational savings of MoE vs. dense layer.

    Args:
        n_experts: Total number of experts
        k_active: Number of active experts (sparsity)
        d_in: Input dimension
        d_out: Output dimension

    Returns:
        Percentage savings in FLOPs
    """
    # pass
    dlflops = n_experts*d_in*d_out
    moeflops = k_active*d_in*d_out
    return round((dlflops-moeflops)*100/dlflops,1)