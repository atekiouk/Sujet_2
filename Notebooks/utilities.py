import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
import re
import sklearn
import random
import textdistance
import umap
import pickle
import spacy.lang.en
from spacy.vocab               import Vocab
from spacy.language            import Language
from spacy.tokens              import Token
from spacymoji                 import Emoji
from sklearn.metrics           import confusion_matrix
from scipy                     import stats
from sklearn.tree              import DecisionTreeClassifier
from sklearn.model_selection   import train_test_split
from sklearn                   import metrics
from sklearn.model_selection   import GridSearchCV
from transformers              import BertTokenizer, BertModel
from sentence_transformers     import SentenceTransformer
from sklearn.metrics           import roc_curve
from umap                      import UMAP
from typing                    import Optional
from sklearn.decomposition     import PCA
from sklearn.linear_model      import LogisticRegression





def t_test(
    dt : pd.DataFrame,
    var : str,
    class_ : str,
          ):
    """
    Calculates the t-test statistic and p-value for the difference in means between two groups, based on the specified variable and class.
Parameters
----------
var : str
    The name of the variable to test.
class_ : str
    The name of the class (group) variable.
Prints
------
t stat : float
    The t-test statistic for the difference in means between the two groups. 
p-value : float
    The p-value for the t-test, indicating the probability of observing the given difference in means by chance.
    """
    t_stat, p_value = stats.ttest_ind(dt[dt[class_]==0][var], dt[dt[class_]==1][var], equal_var=False)
    print(f"t stat : {t_stat}\n\np-value : {p_value}")


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
        corpus[i] = str(corpus[i])
        while(corpus[i][-1]=="#" or corpus[i][-1]=="' '"):
            corpus[i] = corpus[i].rstrip(corpus[i][-1])
    return corpus
            
            
            
            
            
            
            
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
        prefix='has_hashtag'
    elif element == 'emoji':
        detector = _is_emoji
        prefix='has_emoji'
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



def get_presence_of_URL(
    corpus: pd.Series, 
    nlp: spacy.language.Language
) -> pd.Series:
    """
    Detect the presence of URLs in a pandas Series of text .

    Parameters
    ----------
    data : pd.Series
        A pandas Series containing the text to analyze.
    nlp_model : spacy.language.Language
        A pre-trained spacy model.

    Returns
    -------
    pd.Series
        A pandas Series containing 1 if an URL is detected, 0 otherwise.
    """
    has_url = []
    for text in corpus:
        doc = nlp(text)
        found_url = False
        for token in doc:
            if token.like_url:
                found_url = True
                break
        if found_url:
            has_url.append(1)
        else:
            has_url.append(0)
    return pd.Series(has_url)



def get_presence_of_currency_symbol(
    corpus: pd.Series, 
    nlp: spacy.language.Language
) -> pd.Series:
    """
    Detect the presence of currency symbols in a pandas Series of text .

    Parameters
    ----------
    corpus : pd.Series
        A pandas Series containing the text to analyze.
    nlp_model : spacy.language.Language
        A pre-trained spacy model.

    Returns
    -------
    pd.Series
        A pandas Series containing 1 if an URL is detected, 0 otherwise.
    """
    has_currency = []
    for text in corpus:
        doc = nlp(text)
        found_currency = False
        for token in doc:
            if token.is_currency:
                found_currency = True
                break
        if found_currency:
            has_currency.append(1)
        else:
            has_currency.append(0)
    return pd.Series(has_currency)



def get_presence_of_phone_numbers(
    corpus: pd.Series, 
    nlp: spacy.language
) -> pd.Series:
    """
    Detect if a phone number is present in a pandas Series of text .

    Parameters
    ----------
    texts : pandas.core.series.Series
        The pandas series of texts to search for phone numbers.
    nlp : spacy.language
        The spacy model used for text processing.

    Returns
    -------
    pandas.core.series.Series
        A pandas series with name 'has_phone_number', containing 1 if a phone number is present in the corresponding
        text of the input series, and 0 otherwise.
    """
    # Define the regular expression for phone numbers
    regex = r"\+?\d{1,3}[-.\s]?\(?\d{1,3}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}"

    # Compile the regular expression pattern
    pattern = re.compile(regex)

    # Define a function to test if a text contains a phone number
    def has_phone_number(text):
        # Parse the text with the spacy model
        doc = nlp(text)

        # Check if any entity in the document matches the phone number pattern
        for entity in doc.ents:
            if entity.label_ == "PHONE_NUMBER" or pattern.search(entity.text):
                return 1

        # Check if the pattern matches the text directly
        if pattern.search(text):
            return 1

        return 0

    # Apply the has_phone_number function to each text in the input series
    result = corpus.apply(has_phone_number)

    return result


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



def delete_duplicates(
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
    t = [clean_hashtag(s).strip() for s in corpus]
    distance = method #initialisiation de levenshtein avec la distance normalisée.
    i = 0
    r = len(t)
    while(i<r):
        r = len(t)
        j=i+1
        while(j<r):
            if(distance(t[i],t[j]) <= limit ): # Si la distance entre les deux élemens de la liste inf à seuil
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




def plot_confusion_matrix(y_true, y_pred):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Define labels for the classes
    labels = sorted(list(set(y_true)))
    
    # Define figure size and colors
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = sns.color_palette("Blues")
    
    # Create heatmap with annotations
    sns.heatmap(cm, annot=True, cmap=cmap, ax=ax, fmt='g', cbar=False)
    
    # Set labels for axes and title
    ax.set_xlabel('Predicted labels', fontsize=12)
    ax.set_ylabel('True labels', fontsize=12)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Show plot
    plt.show()
    


def plot_roc(y_test : pd.Series,
             y_score : pd.Series,
             color : str
            ) -> None:
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    Parameters
    ----------
    y_test : pd.Series
        True binary labels.
    y_score : pd.Series
        Target scores, can either be probability estimates of the positive class or
        confidence values.
    color : str
        Color for the ROC curve and points.

    Returns
    -------
    None
    """
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = metrics.auc(fpr, tpr)

    # Tracé de la courbe ROC avec seaborn
    sns.set(style='whitegrid', font_scale=1)
    sns.lineplot(x=fpr, y=tpr, color=color, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='orangered', lw=2, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.scatter(fpr, tpr, color=color)
    plt.show()
    
    
def fit_params(
    X: pd.DataFrame,
    y: pd.Series,
    grid_params: dict,
    predictor = DecisionTreeClassifier(),
    hyperopt_params: Optional[dict] = None,
)-> dict:
    """
    Fits a given model on the provided dataset after reducing dimensions with the provided reducer.

    Parameters
    ----------
    X : pd.DataFrame
        The input DataFrame containing the features.
    y : pd.Series
        The target variable.
    grid_params : dict
        The dictionary of hyperparameter grids to search over using GridSearchCV.
    predictor : object, optional
        The model object implementing the scikit-learn estimator interface, by default DecisionTreeClassifier().
    hyperopt_params : dict, optional
        Additional parameters to be passed to the GridSearchCV for hyperparameter optimization, by default None.

    Returns
    -------
    dict
        The best parameters found by the hyperparameter optimization.
    """
    if hyperopt_params is None:
        hyperopt_params = {}
    
    # perform hyperopt with model on reduced data
    return (
        GridSearchCV(
            estimator=predictor,
            param_grid=grid_params,
            **hyperopt_params
        )
        .fit(X, y)
        .best_params_
    )
    
    
def reducing_grid_s(X_train: pd.DataFrame,
                X_test: pd.DataFrame,
                y_train: pd.DataFrame, 
                y_test: pd.DataFrame,
                reduc: object,
                predic: object,
                grid_params: dict
               )-> tuple[object, float]:
    """
    Perform dimensionality reduction and grid search on a given dataset using a specified reducer and predictor.

    Parameters:
        X_train (pd.DataFrame): Training data features.
        X_test (pd.DataFrame): Testing data features.
        y_train (pd.DataFrame): Training data labels.
        y_test (pd.DataFrame): Testing data labels.
        reduc (object): Dimensionality reduction model (e.g., PCA, UMAP).
        predic (object): Predictor model (e.g., DecisionTreeClassifier, LogisticRegression).
        grid_params (dict): Grid search parameters for the predictor.

    Returns:
        tuple[object, float]: Trained model and its corresponding AUC score.
    """
    hyperopt_params={
                "scoring": "roc_auc",
                "cv": 5,
                "refit": True}
    reducer = reduc.fit(X_train)
    predictor = predic
    X_train_reduced = reducer.transform(X_train)
    X_test_reduced = reducer.transform(X_test)
    opt_params = fit_params(
            X=X_train_reduced,
            y=y_train,
            predictor=predictor,
            grid_params=grid_params,
            hyperopt_params = hyperopt_params)
    model = predictor.set_params(**opt_params).fit(X_train_reduced, y_train)
    y_pred = model.predict(X_test_reduced)
    y_score = model.predict_proba(X_test_reduced)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_score)
    auc = metrics.auc(fpr, tpr)
    return model,auc
    
    
def auto_model_selection(
X : pd.Series,
y : pd.Series,
dimension_range : range
)-> tuple[object,float,int,object]:
    """
    Automatically selects the best model using dimensionality reduction techniques (PCA, UMAP)
    and evaluates their performance based on ROC AUC score.

    Parameters
    ----------
    X : pd.Series
        The input feature data.
    y : pd.Series
        The target variable.
    dimension_range : range
        The range of dimensions to explore for dimensionality reduction.

    Returns
    -------
    Tuple[object, float, int]
        A tuple containing the best model, the best ROC AUC score, and the optimal number of dimensions.
    """
    best_auc = 0
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    grid_params_lr = {
                'solver': ['lbfgs', 'liblinear', 'newton-cg','newton-cholesky','sag','saga']
            }
    grid_params_tree = {
                'criterion': ['entropy', 'gini', 'log_loss'],
                'max_depth' : np.arange(2, 6, dtype=int),
                'ccp_alpha' : np.linspace(0.0, 0.20, 5)
            }
    auc_scores = {}

    for dim in dimension_range:
        for pred_grid in [{'predictor' : DecisionTreeClassifier(random_state=42),'grid' : grid_params_tree}, {'predictor' : LogisticRegression(random_state = 42,max_iter=1000), 'grid' : grid_params_lr}]:
            for red in [UMAP(n_components = dim), PCA(n_components = dim)]:
                model,auc = reducing_grid_s(
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train, 
                    y_test=y_test,
                    reduc=red,
                    predic=pred_grid['predictor'],
                    grid_params=pred_grid['grid']
                   )
                if red.__class__.__name__ not in auc_scores:
                    auc_scores[red.__class__.__name__] = {}
                if pred_grid['predictor'].__class__.__name__ not in auc_scores[red.__class__.__name__]:
                    auc_scores[red.__class__.__name__][pred_grid['predictor'].__class__.__name__] = []

                auc_scores[red.__class__.__name__][pred_grid['predictor'].__class__.__name__].append(auc)
                if(auc>best_auc):
                    best_auc = auc
                    best_model = model
                    opt_nb_dim = dim
                    reducer = red

    plt.figure(figsize=(10, 6))
    for reducer_, predictor_scores in auc_scores.items():
        for predictor_, scores in predictor_scores.items():
            plt.plot(list(dimension_range), scores, label=f"{reducer_} + {predictor_}")

    plt.xlabel("Number of Dimensions")
    plt.ylabel("AUC Score")
    plt.title("AUC Score vs. Number of Dimensions")
    plt.legend()
    plt.show()

    return best_model,best_auc,opt_nb_dim,reducer



def evaluate(
    y_test: pd.Series,
    y_pred: pd.Series,
    y_score: pd.Series,
) -> None:
    """
    Evaluate the performance of a binary classification model by computing various metrics and plotting the confusion matrix.

    Parameters
    ----------
    y_test : pd.Series
        The true labels of the test set.
    y_pred : pd.Series
        The predicted labels of the test set.
    y_score : pd.Series
        The predicted scores or probabilities of the positive class for the test set.

    Returns
    -------
    None

    Prints
    ------
    AUC : float
        The Area Under the ROC Curve (AUC) score.
    Accuracy score : float
        The accuracy score, which measures the proportion of correct predictions.
    Precision score : float
        The precision score, which measures the ability of the model to correctly identify positive samples.
    Recall score : float
        The recall score, which measures the ability of the model to correctly identify positive samples among all true positive samples.

    Displays
    --------
    Confusion Matrix : matplotlib.pyplot figure
        A visual representation of the confusion matrix, showing the true and predicted labels.

    """
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
    print(f"AUC : {metrics.auc(fpr, tpr):.3f}")
    print(f"Accuracy score : {metrics.accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision score : {metrics.precision_score(y_test, y_pred):.3f}")
    print(f"Recall score : {metrics.recall_score(y_test, y_pred):.3f}")
    plot_confusion_matrix(y_true=y_test, y_pred=y_pred)
    
    
    
def junk_classifier(
Corpus_instagram : pd.DataFrame,
model : object,
reducer : object
)-> pd.DataFrame:
    """
    Applies junk classification on the given Instagram corpus using logistic regression.

    Parameters:
        Corpus_instagram (pd.DataFrame): The Instagram corpus containing the text data.
        model : The sklearn model we will use to classify the Instagram publications.
        reducer  : Dimensionality reduction model (e.g., PCA, UMAP).
    Returns:
        pd.DataFrame: Subset of Corpus_instagram containing only the predicted non junk entries.
    """
    #delete exact same or similar publication
    dist = textdistance.levenshtein.normalized_distance
    Corpus_instagram = delete_duplicates(Corpus_instagram['text'].tolist(),Corpus_instagram['publication_time'].tolist(),0.6,dist)
    
    #Encode corpus
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    Corpus_encoded = encoder.encode(Corpus_instagram['text'].tolist())
    
    #Reduce to 59 dimensions using UMAP
    Corpus_encoded_reduced =  reducer.transform(Corpus_encoded)
    
    #Predict junk publication among the corpus
    proba_threshold = 0.5
    y_pred = np.where(model.predict_proba(Corpus_encoded_reduced)[:, 1] > proba_threshold, 1, 0)
    
    return Corpus_instagram[y_pred==1]