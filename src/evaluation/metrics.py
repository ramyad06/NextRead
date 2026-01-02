import numpy as np

def precision_at_k(recommended_indices, true_indices, k):
    """
    Calculates Precision@K.
    recommended_indices: list of article indices recommended by model
    true_indices: set or list of article indices relevant to the user (ground truth)
    k: cutoff
    """
    if not true_indices:
        return 0.0
        
    recommended_k = recommended_indices[:k]
    hits = 0
    for idx in recommended_k:
        if idx in true_indices:
            hits += 1
            
    return hits / k

def ndcg_at_k(recommended_indices, true_indices, k):
    """
    Calculates NDCG@K.
    Assuming binary relevance (1 if in true_indices, 0 otherwise).
    """
    if not true_indices:
        return 0.0
        
    recommended_k = recommended_indices[:k]
    dcg = 0.0
    for i, idx in enumerate(recommended_k):
        if idx in true_indices:
            # log2(i+2) because i is 0-indexed (rank 1 is i=0)
            dcg += 1.0 / np.log2(i + 2)
            
    # Calculate IDCG
    # Best possible ordering has all true items at the top
    num_true = len(true_indices)
    idcg = 0.0
    for i in range(min(num_true, k)):
        idcg += 1.0 / np.log2(i + 2)
        
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg

def calculate_ctr(hits, impressions):
    if impressions == 0:
        return 0.0
    return hits / impressions
