import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix


def synthetic_adata(
    n_cells=3000,
    n_genes=2000,
    n_cell_types=3,
    n_de_genes=100,
    fc_range=(1.1, 3.0),
    mean_counts=10,
    dispersion=0.5,
    dropout_rate=0.5,
    n_samples=6,
    random_seed=42,
):
    """
    Create a synthetic AnnData object with simulated differential expression using negative binomial distribution.

    Parameters
    ----------
    n_cells : int
        Number of cells to simulate
    n_genes : int
        Number of genes to simulate
    n_cell_types : int
        Number of different cell types
    n_de_genes : int
        Number of genes that are differentially expressed
    fc_range : float
        Max fold change for differentially expressed genes
    mean_counts : float
        Mean number of counts per gene
    dispersion : float
        Dispersion parameter for negative binomial (higher = more variance)
    dropout_rate : float
        Base dropout rate (probability of a count becoming zero)
    n_samples : int
        Number of independent samples per condition
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    adata : AnnData
        Synthetic AnnData object with simulated count data and log-normalized data
    """
    np.random.seed(random_seed)

    # Create cell type labels
    cell_types = [f"cell_type_{i}" for i in range(n_cell_types)]
    cell_type_labels = np.random.choice(cell_types, size=n_cells)

    # Create sample labels (6 samples per condition)
    samples = [f"sample_{i}" for i in range(n_samples)]
    sample_labels = np.random.choice(samples, size=n_cells)

    # Create condition labels (control vs treatment)
    conditions = np.random.choice(["control", "treatment"], size=n_cells)

    # Create gene names
    gene_names = [f"gene_{i}" for i in range(n_genes)]

    # Initialize expression matrix
    X = np.zeros((n_cells, n_genes))

    def apply_dropout(counts, mean_expression):
        """Apply dropout based on mean expression level"""
        # Higher dropout probability for lower expressed genes
        dropout_prob = dropout_rate * np.exp(-mean_expression / mean_counts)
        dropout_mask = np.random.random(counts.shape) < dropout_prob
        counts[dropout_mask] = 0
        return counts

    # Generate base expression for each cell type
    for cell_type in cell_types:
        cell_type_mask = cell_type_labels == cell_type
        # Generate cell-type specific mean expression
        base_expression = np.random.gamma(shape=2, scale=mean_counts, size=n_genes)

        # Generate negative binomial counts for this cell type
        for gene in range(n_genes):
            # Calculate p parameter for negative binomial
            p = 1 / (1 + dispersion * base_expression[gene])
            # Generate counts
            counts = np.random.negative_binomial(n=1 / dispersion, p=p, size=np.sum(cell_type_mask))
            # Apply dropout
            counts = apply_dropout(counts, base_expression[gene])
            X[cell_type_mask, gene] = counts

    # Add differential expression for selected genes
    de_genes = np.random.choice(n_genes, size=n_de_genes, replace=False)
    for gene in de_genes:
        # Add fold change for treatment condition
        treatment_mask = conditions == "treatment"
        # For treatment cells, multiply the mean by fold change and regenerate counts
        de_fold_change = np.random.uniform(fc_range[0], fc_range[1])
        de_direction = np.random.choice([-1, 1])
        de_fold_change = 1 / de_fold_change if de_direction == -1 else de_fold_change
        # Calculate new mean expression for treatment cells
        base_expression = X[treatment_mask, gene].mean() * de_fold_change
        p = 1 / (1 + dispersion * base_expression)
        counts = np.random.negative_binomial(n=1 / dispersion, p=p, size=np.sum(treatment_mask))
        # Apply dropout
        counts = apply_dropout(counts, base_expression)
        X[treatment_mask, gene] = counts

    # Create AnnData object
    adata = sc.AnnData(csr_matrix(X.astype(int)))
    adata.var_names = gene_names
    adata.obs["cell_type"] = cell_type_labels
    adata.obs["condition"] = conditions
    adata.obs["sample"] = sample_labels
    adata.obs["condition_sample"] = adata.obs["condition"] + "_" + adata.obs["sample"]

    # Add some metadata
    adata.uns["de_genes"] = np.array(gene_names)[de_genes]
    adata.uns["de_fold_change"] = de_fold_change
    adata.uns["mean_counts"] = mean_counts
    adata.uns["dispersion"] = dispersion
    adata.uns["dropout_rate"] = dropout_rate

    # Store raw counts
    adata.layers["counts"] = adata.X.copy()
    adata.raw = adata.copy()

    # Normalize and log transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata
