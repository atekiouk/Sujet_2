import pandas as pd
import numpy as np
import spacy


def create_dummies(
    corpus: pd.Series,
    element: str,
    top: int = 5
) -> pd.DataFrame:
    """
    Create dummy encodings for most frequents text elements in the given corpus.
    
    Parameters
    ----------
    corpus: pd.Series
        The corpus on which the elements will be searched for.
    element: str, {'hashtag', 'emoji'}
        The text element to look for. Currently, only 'hashtag' and 'emoji' are supported.
    top: int, defaults to 5
        The number of top modalities to dummy encode.
    
    Returns
    -------
    pd.DataFrame
        The dummy encoding corresponding to the most frequent modalities of the specified elements.
    """
    if element == 'hashtag':
        regex = r'\#\w+'
    elif element == 'emoji':
        regex = r'[^\w\s,]'
    else:
        raise ValueError("Only 'hashtag' and 'emoji' elements are supported.")

    all_elements = corpus.str.findall(regex).explode()
    top_elements = all_elements.value_counts().head(top).index

    dummy = pd.DataFrame(index=corpus.index)
    for e in top_elements:
        dummy[e] = corpus.str.contains(e).astype(int)
    
    return dummy