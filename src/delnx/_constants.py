SUPPORTED_BACKENDS = ["jax", "statsmodels", "cuml", "pydeseq2"]

COMPATIBLE_DATA_TYPES = {
    "deseq2": ["counts"],
    "negbinom": ["counts"],
    "binomial": ["binary"],
    "lr": ["lognorm", "binary", "scaled"],
    "anova": ["lognorm", "scaled"],
    "anova_residual": ["lognorm", "scaled"],
}
