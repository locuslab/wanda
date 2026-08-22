def format_result_log(method, sparsity_ratio, perplexity, pruning_time_seconds):
    return (
        "method\tactual_sparsity\tppl_test\tpruning_time_seconds\n"
        f"{method}\t{sparsity_ratio:.4f}\t{perplexity:.4f}\t{pruning_time_seconds:.2f}\n"
    )
