import pandas as pd
import numpy as np
import spacy
import re
import spacy.lang.en
from spacy.vocab               import Vocab
from spacy.language            import Language
from spacy.tokens              import Token
from spacymoji                 import Emoji


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



def clean_hashtag(text: str) -> str:
    """
    Remove hashtags from a text.

    Parameters
    ----------
    text : str
        The input text.

    Returns
    -------
    str
    """
    hashtag_pattern= re.compile("#[A-Za-z0-9_]+")
    return re.sub(hashtag_pattern,"", text) #On supprime tout les types de #
    
    
def clear_trailing_hash(
    corpus: pd.Series
) -> None:
    """
    Clear any trailing '#' character from each string in a list.
    
    Parameters
    ----------
    corpus : list of str
        pd.Series of strings to be processed.
    
    Returns
    -------
    None
        The function only modifies the input corpus list in place.
    """
    for i in range(len(corpus)):
        if(corpus[i][-1]=="#"):
            corpus[i] = corpus[i].rstrip(corpus[i][-1])
            
            
            
            
            
            
            
            
# TODO: voir si l'on peut passer en paramètre un objet spacy qui identifie le type de token,
# afin de n'écrire qu'une seule méthode (pour les hashtags, les emojis, etc...).
def top_hashtags(
    corpus: pd.Series,
    nlp : spacy.lang.en.English,
    top: int = 5
) -> pd.Series:
    """
    Retrieves the most frequent hashtags in the given corpus.

    Parameters
    ----------
    corpus : pd.Series
        The pd.Series of text documents to retrieve hashtags from.
    top : int, optional
        The number of top hashtags to return. Default is 5.
    nlp : spacy.lang.en.English
        A spacy language model with a custom pipe to detect hashtags.

    Returns
    -------
    pd.Series
        A pandas Series containing the count of each of the most frequent hashtags found in the corpus,
        sorted in descending order.
    """
    # retrieve all hashtags in corpus
    hashtags = []
    for i in corpus:
        doc = nlp(i)
        for token in doc:
            if token._.is_hashtag:
                hashtags.append(token.text)
    # count hashtags & return most frequents
    return (
        pd
        .Series(hashtags)
        .value_counts()
        .sort_values(ascending=False)
        .head(top)
    )







def top_emojis(
    corpus: pd.Series,
    nlp : spacy.lang.en.English,
    top: int = 5
 ) -> pd.Series:
    """
    Retrieves the most frequent emojis in the given corpus.
    
    Parameters
    ----------
    corpus : pd.Series
        The pd.Series of text documents to retrieve hashtags from.
    top : int, optional
        The number of top hashtags to return. Default is 5.
    nlp : spacy.lang.en.English
        A spacy language model with a custom pipe to detect hashtags.
    
    Returns
    -------
    pd.Series
        A pandas Series containing the count of each of the most frequent emojis found in the corpus,
        sorted in descending order.
    """
    # Retrieve emojis in corpus
    emojis = []
    for i in corpus:
        doc = nlp(i)
        for token in doc:
            if token._.is_emoji:
                emojis.append(token.text)

    # count occurrences & return most frequent
    return (
        pd
        .Series(emojis)
        .value_counts()
        .sort_values(ascending=False)
        .head(top)
    )



def create_dummies(corpus: pd.Series,
                   y : pd.Series,
                   element: str,
                   nlp : spacy.lang.en.English, 
                   top: int = 5
) -> pd.DataFrame:
    """
    Create dummy encodings for most frequents text elements in the given corpus.

    Parameters
    ----------
    corpus : pd.Series
        The corpus on which the elements will be searched for.
    y : pd.Series
        The variable which will enable to filter the rows to use to determine the top elements+
    element : str, {'hashtag', 'emoji'}
        The text element to look for. Currently, only 'hashtag' and 'emoji' are supported.
    nlp : spacy.lang.en.English
        A spacy language model with a custom pipe to detect hashtags and emojis.
    top : int, optional
        The number of top modalities to dummy encode. Default is 5.

    Returns
    -------
    pd.DataFrame
        The dummy encoding corresponding to the most frequent modalities of the specified elements.
    
    Raises
    ------
    ValueError: if ``element`` is not supported.
    """
    def _is_hashtag(token):
        return token._.is_hashtag
    def _is_emoji(token):
        return token._.is_emoji
    if element == 'hashtag':
        detector = _is_hashtag
    elif element == 'emoji':
        detector = _is_emoji
    else:
        raise ValueError("Only 'hashtag' and 'emoji' elements are supported.")

    top_elements = (
        corpus[y==1]
        .apply(lambda text: [token.text for token in nlp(text) if detector(token)])
        .explode()
        .value_counts()
        .head(top)
        .index
    )

    dummy = pd.DataFrame(index=corpus.index)
    for e in top_elements:
        dummy[e] = corpus.apply(lambda text: 1 if e in [token.text for token in nlp(text) if detector(token)] else 0)

    return dummy



def get_word_ratio(
    corpus: list, 
    nlp: spacy.lang.en.English
) -> list[float]:
    """
    Computes the ratio of words to hashtags in the text column of the given dataframe.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe containing the text column.
    nlp: spacy.lang.en.English
        The spacy English pipeline.

    Returns
    -------
    List[float]
        The list of word-to-hashtag ratios computed for each text in the dataframe.
    """
    ratio = []
    for i in corpus:
        doc = nlp(i)
        nb_word = 0
        nb_hash = 0
        for token in doc:
            if(token._.is_hashtag):
                nb_hash+=1
            else:
                nb_word+=1
        if((nb_hash+nb_word)!=0):
            ratio.append(nb_word/(nb_hash+nb_word))
        else:
            ratio.append(0)  
    return ratio



def get_caps_ratio(
    corpus: list,
    nlp: spacy.lang.en.English
) -> list[float]:
    """
    Calculates the ratio of capitalized words in each text of the given DataFrame.
    
    Parameters
    ----------
    corpus: list
        The list containing the text to analyze.
    
    Returns
    -------
    list[float]
        The list containing the ratios of capitalized words for each text.
    """
    ratio = []
    for i in corpus:
        doc = nlp(clean_hashtag(i))
        nb_lower = 0
        nb_caps = 0
        for token in doc:
            if(token.text.isupper()):
                nb_caps+=1
            else:
                nb_lower+=1
        if((nb_caps+nb_lower)!=0):
            ratio.append(nb_caps/(nb_caps+nb_lower))
        else:
            ratio.append(1)
    return ratio



def get_normalized_nb_punct(
        corpus: list,
        nlp: spacy.lang.en.English
        ) -> list:
    """
    Counts the number of punctuation symbols in each string of the given corpus.
    
    Parameters
    ----------
    corpus : list
        A list of strings to be processed.
    nlp : spacy.lang.en.English
        A spaCy English language processing pipeline instance.

    Returns
    -------
    list
        A list containing the number of punctuation symbols in each string of the corpus.
    """
    tot = []
    for i in corpus:
        doc = nlp(clean_hashtag(i))
        nb_punct = 0
        for token in doc:
            if(token.is_punct):
                nb_punct+=1
        if(len(doc)!=0):
            tot.append(nb_punct/len(doc))
        else:
            tot.append(1)
    return tot



def del_double(
    corpus: list, 
    publication_time: list, 
    limit: float, 
    method: callable
    ) -> list: 
    """
    Remove duplicated elements from a list of strings using Levenshtein distance.

    Parameters
    ----------
    corpus : list of str
        The list of strings to remove duplicates from.
    publication_time : list of timestamp
        The list of publication time for each element in the corpus.
    limit : float
        The distance threshold under which two elements are considered duplicates.
        Must be in the range [0, 1] if using normalized Levenshtein distance, or
        in the range [0, 100] if using classical Levenshtein distance.
    method : function
        The method to use to compute the distance between two strings.

    Returns
    -------
    list of str
        The list of strings with duplicates removed.
    """
    
    t = corpus.copy()
    distance = method #initialisiation de levenshtein avec la distance normalisée.
    i = 0
    r = len(t)
    while(i<r):
        r = len(t)
        j=i+1
        while(j<r):
            if(distance(clean_hashtag(t[i]).strip(),clean_hashtag(t[j]).strip()) <= limit ): # Si la distance entre les deux élemens de la liste inf à seuil
                if(publication_time[i]<publication_time[j]):
                    del t[j] #delete
                    r = len(t) #on actualise la taille de la liste
                else:
                    del t[i]
                    r = len(t) #on actualise la taille de la liste
            else:
                j+=1
        i+=1
    return t



def train_test(
    data: pd.DataFrame,
    y: str,
    f_y0: float, 
    f_y1: float
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a pandas DataFrame into two DataFrames for training and testing purposes.

    Parameters:
    ----------
        data : pd.DataFrame
            The input DataFrame.
        y : str 
            The target variable.
        f_y0 : float
            The fraction of rows with y=0 to include in the training DataFrame.
        f_y1 : float
            The fraction of rows with y=1 to include in the training DataFrame.

    Returns:
        tuple(pd.DataFrame, pd.DataFrame): The training and testing DataFrames.
    """
    num_y0 = (data[y] == 0).sum()
    num_y1 = (data[y] == 1).sum()
    
    # Calculate the number of rows to put in train set and test set
    num_train_y0 = int(num_y0 * f_y0)
    num_train_y1 = int(num_y1 * f_y1)
    num_train = num_train_y0 + num_train_y1
    
    # Split the data into train and test sets
    train_y0 = data[data[y] == 0].sample(n=num_train_y0)
    train_y1 = data[data[y] == 1].sample(n=num_train_y1)
    train = pd.concat([train_y0, train_y1])
    test = data.drop(train.index)
    
    return train, test