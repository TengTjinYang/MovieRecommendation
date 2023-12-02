import information_retrieval
import numpy as np

def precision_at_k(r, k):
    """
    Score is precision @ k (This we solve for you!)
    Relevance is binary (nonzero is relevant).
    
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
    File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    
    Args:
    r: Relevance scores (list or numpy) in rank order
       (first element is the first item)
    
    Returns:
    Precision @ k
    
    Raises:
    ValueError: len(r) must be >= k
    """
    
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

def average_precision(r):
    """
    Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    
    Args:
    r: Relevance scores (list or numpy) in rank order
       (first element is the first item)
    
    Returns:
    Average precision
    """

    rel_positions = [i for i, rel in enumerate(r) if rel]
    if not rel_positions:
        return 0
    avg_p = sum([precision_at_k(r, k+1) for k in rel_positions]) / len(rel_positions)
    return avg_p

def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values. Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per
    .pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
    r: Relevance scores (list or numpy) in rank order
    (first element is the first item)
    k: Number of results to consider
    method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307,
    ...]
    If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
    Discounted cumulative gain
    """
    #write your code here (return appropriate value)
    r = (r[:k])
    if not r:
        return 0.
    if method == 0:
        return r[0] + sum(val / np.log2(i + 2) for i, val in enumerate(r[1:]))
    elif method == 1:
        return sum(val / np.log2(i + 2) for i, val in enumerate(r))
    else:
        raise ValueError('Invalid method: {}'.format(method))

def mean_average_precision(rs):
    """
    Score is mean average precision
    Relevance is binary (nonzero is relevant).

    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666

    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order (first element is the first item)
    
    Returns:
        Mean average precision
    """
    m_avg_p = sum(average_precision(r) for r in rs) / len(rs)
    
    return m_avg_p


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values. Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per
    .pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
    r: Relevance scores (list or numpy) in rank order
    (first element is the first item)
    k: Number of results to consider
    method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307,
    ...]
    If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
    Normalized discounted cumulative gain
    """
    #write your code here (return appropriate value)
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

import pandas as pd

# Load the files into pandas DataFrames
basics_df = pd.read_csv('Database/testImdbTitleBasicsTest.csv')

def get_movie_name_by_tconst(tconst):
    # Extract the row with the given 'tconst'
    movie_row = basics_df[basics_df['tconst'] == tconst]
    
    # Check if the movie exists in the DataFrame
    if movie_row.empty:
        return "Movie with the given tconst not found."
    
    # Extract the movie's primary title
    movie_name = movie_row.iloc[0]['primaryTitle']
    return movie_name
