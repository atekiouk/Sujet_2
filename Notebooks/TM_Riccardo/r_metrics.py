import pandas as pd
import plotly.graph_objs as go
import numpy as np
import re
from gensim.models.coherencemodel import CoherenceModel
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from gensim import corpora, models
import gensim
import matplotlib.pyplot as plt
import math


def compute_coherence(docs, topic_words, metric= 'c_v', window_size = 10):
    """
    This function computes the coherence score of a list of topic words for a given set of documents.

    Parameters:
    - docs (list): A list of strings representing the documents to evaluate.
    - topic_words (list): A list of lists containing the top words for each topic. Each inner list contains the top words for a single topic, in descending order of importance.
    - metric (str, optional): The coherence metric to use. Defaults to 'c_v'. Possible values are 'c_v', 'u_mass', 'c_uci', 'c_npmi'.
    
    Returns:
    - coherence (float): The coherence score of the topic words for the given set of documents.
    
    The function first preprocesses the documents by removing newline and tab characters, removing non-alphanumeric characters, and replacing empty 
    documents with the string "emptydoc".

    It then constructs a bag-of-words representation of the documents using the CountVectorizer class from scikit-learn. 
    The resulting features are used to construct a dictionary and a corpus for the CoherenceModel class from the gensim library.

    Finally, the coherence of the topic words is computed using the specified coherence metric and returned as a float value.
    """
    
    def preprocess_text(documents):
        cleaned_documents = [doc.replace("\n", " ") for doc in documents]
        cleaned_documents = [doc.replace("\t", " ") for doc in cleaned_documents]
        cleaned_documents = [re.sub(r'[^A-Za-z0-9 ]+', '', doc) for doc in cleaned_documents]
        cleaned_documents = [doc if doc != "" else "emptydoc" for doc in cleaned_documents]
        return cleaned_documents
    
    cleaned_docs = preprocess_text(docs)
    vectorizer = CountVectorizer(stop_words="english").fit(cleaned_docs)
    tokenizer = vectorizer.build_tokenizer()

    # Extract features for Topic Coherence evaluation
    words = vectorizer.get_feature_names_out()

    tokens = [tokenizer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)

    corpus = [dictionary.doc2bow(token) for token in tokens]

    # Evaluate
    coherence_model = CoherenceModel(topics=topic_words, 
                                     texts=tokens, 
                                     corpus=corpus,
                                     dictionary=dictionary, 
                                     coherence=metric,
                                    window_size = window_size)
    
    coherence = coherence_model.get_coherence_per_topic()
    return coherence


def compute_topic_diversity(words_list):
    """
    Computes the topic diversity percentage of a list of words lists.

    Args:
        words_list (list): A list of lists of words.

    Returns:
        float: The percentage of unique words in the flattened list of words lists.
    """
    flattened_list = [item for sublist in words_list for item in sublist]
    unique_words = set(flattened_list)
    percentage = len(unique_words) / len(flattened_list) * 100
    return percentage


def compute_similarity_intra_cluster(df, model_name, sentence_transformer = 'all-MiniLM-L6-v2', embeddings=None):
    """
    Computes the mean and standard deviation of the cosine similarity matrix between documents within each topic cluster
    in a DataFrame.

    Args:
    - df: a pandas DataFrame containing a 'post_final' column with preprocessed text documents and a column with the 
          topic cluster labels for each document (specified by the `model_name` argument).
    - model_name: a string specifying the name of the column containing the topic cluster labels in the DataFrame.
    - sentence_transformer: a string specifying the name of the SentenceTransformer model to use for generating 
                            document embeddings. Default is 'all-MiniLM-L6-v2'.
    - embeddings: an optional numpy array of precomputed document embeddings. If provided, these will be used instead 
                  of computing new embeddings using the SentenceTransformer model.

    Returns:
    - A tuple of two 1-D numpy arrays:
        - similarities_mean: an array of length `n_topics` containing the mean cosine similarity values between 
                             documents within each topic cluster.
        - similarities_std: an array of length `n_topics` containing the standard deviation of the cosine similarity 
                            values between documents within each topic cluster.
    """
    if embeddings is None:
        # Initialize a SentenceTransformer model
        model = SentenceTransformer(sentence_transformer)

        # Get sentence embeddings for all documents
        embeddings = model.encode(df['post_final'].tolist())
    
    df["embeddings"] = embeddings.tolist()
    nr_topics = len(set(df[model_name]))
    
    if model_name.startswith("bert"):
        nr_topics -= 1

        
    similarities_mean = np.zeros(nr_topics)
    similarities_std = np.zeros(nr_topics)
    for i in range(nr_topics):
        # Get embeddings for documents in this cluster
        cluster_embeddings = np.vstack(df[df[model_name]==i]["embeddings"])

        # Compute cosine similarity matrix for this cluster
        sim_matrix = cosine_similarity(cluster_embeddings)

        # Compute mean cosine similarity for this cluster
        similarities_mean[i] = np.mean(sim_matrix)
        similarities_std[i] = np.std(sim_matrix)

    return similarities_mean, similarities_std


def compute_cosine_similarity_words(words_lists, sentence_transformer = 'all-MiniLM-L6-v2', w2v = False):
    """
    Computes the average cosine similarity between all pairs of words in each list of words provided in the `words_lists`
    argument, using either a pre-trained sentence transformer model or a pre-trained word2vec model.

    Args:
    - words_lists: A list of lists, where each inner list contains the words to be compared.
    - sentence_transformer: (Optional) A string representing the name of the pre-trained sentence transformer model to use.
      Default is 'all-MiniLM-L6-v2'.
    - w2v: (Optional) A boolean indicating whether to use a pre-trained word2vec model instead of the sentence transformer.
      Default is False.

    Returns:
    - A list of float values representing the average cosine similarity between all pairs of words in each list of words
      provided in the `words_lists` argument.
    """
    
    
    def compute_cos_sim(words_list):
        """
        Computes the average cosine similarity between all pairs of words in the `words_list` argument, using either a
        pre-trained sentence transformer model or a pre-trained word2vec model.

        Args:
        - words_list: A list of words to be compared.

        Returns:
        - A float value representing the average cosine similarity between all pairs of words in the `words_list`
          argument.
        """
        similarity_list = []
        for i in range(len(words_list)):
            for j in range(i+1, len(words_list)):
                # Encode the two words
                if w2v == True:
                    if words_list[i] in model.key_to_index  and words_list[j] in model.key_to_index:
                        v1 = model[words_list[i]]
                        v2 = model[words_list[j]]
                        sim = cosine_similarity(v1.reshape(1,-1), v2.reshape(1,-1))[0][0]
                        similarity_list.append(sim)
                else:
                    v1 = sentence_model.encode(words_list[i])
                    v2 = sentence_model.encode(words_list[j])
                    sim = cosine_similarity(v1.reshape(1,-1), v2.reshape(1,-1))[0][0]
                    similarity_list.append(sim)
               
        return np.mean(similarity_list)
    
    sentence_model = SentenceTransformer(sentence_transformer)
    model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
    similarity = []
    for words_list in words_lists:
        similarity.append(compute_cos_sim(words_list))
            
    return similarity


def plot_metrics(lists_of_metrics, model_names, title, y_label, ylim=(0,1)):
    num_topics = len(lists_of_metrics[0])
    x = range(1, num_topics+1)

    for i in range(len(lists_of_metrics)):
        mean_cv = np.mean(lists_of_metrics[i])
        plt.axhline(y=mean_cv, color=['r', 'g', 'b'][i], linestyle='--') #,label=f"{model_names[i]} mean")

        plt.scatter(x, lists_of_metrics[i], c=['r', 'g', 'b'][i], marker=['o', 's', '^'][i], label=model_names[i])

    plt.ylim(ylim)
    plt.xticks(x)
    plt.xlabel('topic')
    plt.ylabel(y_label)
    plt.suptitle(title)
    plt.title('(topics plotted by ascending values -> topics do not match between models)', fontsize=10)
    plt.legend()
    plt.show()
    
    
def plot_diversities(diversities, model_names):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(diversities))

    ax.bar(x, diversities, color=['r', 'g', 'b', 'c', 'm', 'y', 'k'])
    ax.set_xticks(x)
    ax.set_xticklabels(model_names,ha='right')
    plt.ylim((0,100))
    ax.set_xlabel('Model')
    ax.set_ylabel('Topic Diversity')
    ax.set_title('Topic Diversities of Different Models')

    plt.show()
    
    


# Define a function to calculate the circle packing density for a DataFrame of circles
def circle_packing_density(circles_df):
    """
    Calculate the circle packing density for a DataFrame of circles.

    Parameters:
    circles_df (pandas.DataFrame): A DataFrame containing information about circles, including the x and y coordinates of the circle
    center and the circle radius.

    Returns:
    float: The circle packing density, which is defined as the total area of overlap between all pairs of circles in the DataFrame 
    divided by the total area of all the circles in the DataFrame.

    Example:
    >>> import pandas as pd
    >>> circles = pd.DataFrame({'x': [0, 1, 2], 'y': [0, 0, 0], 'radius': [1, 1, 1]})
    >>> circle_packing_density(circles)
    0.5235987755982989
    """
        # Define a function to calculate the area of overlap between two circles
    def circle_intersection_area(c1, c2):
        """
        Calculate the area of overlap between two circles.

        Parameters:
        -----------
        c1 : dict
            Dictionary representing the first circle with keys: 'x', 'y', 'radius'.
        c2 : dict
            Dictionary representing the second circle with keys: 'x', 'y', 'radius'.

        Returns:
        --------
        float
            The area of overlap between the two circles.
        """
        d = math.sqrt((c1['x'] - c2['x'])**2 + (c1['y'] - c2['y'])**2) # distance between centers
        if d > c1['radius'] + c2['radius']: # circles do not overlap
            return 0
        elif d < abs(c1['radius'] - c2['radius']): # one circle inside the other
            return math.pi * min(c1['radius'], c2['radius'])**2
        else: # circles partially overlap
            alpha = math.acos((c1['radius']**2 + d**2 - c2['radius']**2) / (2 * c1['radius'] * d))
            beta = math.acos((c2['radius']**2 + d**2 - c1['radius']**2) / (2 * c2['radius'] * d))
            a1 = c1['radius']**2 * alpha - c1['radius']**2 * math.sin(alpha * 2) / 2
            a2 = c2['radius']**2 * beta - c2['radius']**2 * math.sin(beta * 2) / 2
            return a1 + a2

    # Define a function to calculate the total area of overlap between all pairs of circles in a DataFrame
    def total_overlap_area(circles_df):
        """
        Calculates the total area of overlap between all pairs of circles in the given DataFrame.

        Parameters:
        circles_df (pandas DataFrame): A DataFrame containing the circles to calculate the total overlap area for. The DataFrame 
        should have the following columns:
            - 'x': The x-coordinate of the center of the circle.
            - 'y': The y-coordinate of the center of the circle.
            - 'radius': The radius of the circle.

        Returns:
        float: The total area of overlap between all pairs of circles in the DataFrame.
        """
        total_overlap_area = 0
        for i in range(len(circles_df)):
            for j in range(i + 1, len(circles_df)):
                total_overlap_area += circle_intersection_area(circles_df.iloc[i], circles_df.iloc[j])
        return total_overlap_area

    # Define a function to calculate the area of the bounding rectangle that encloses all the circles in a DataFrame
    def bounding_rect_area(circles_df):
        """
        Calculates the area of the smallest rectangle that encloses all the circles in the given DataFrame.

        Parameters:
        circles_df (pandas DataFrame): A DataFrame containing the circles to calculate the bounding rectangle area for. The
        DataFrame should have the following columns:
            - 'x': The x-coordinate of the center of the circle.
            - 'y': The y-coordinate of the center of the circle.
            - 'radius': The radius of the circle.

        Returns:
        float: The area of the smallest rectangle that encloses all the circles in the DataFrame.
        """
        min_x = circles_df['x'].min() - circles_df['radius'].max()
        max_x = circles_df['x'].max() + circles_df['radius'].max()
        min_y = circles_df['y'].min() - circles_df['radius'].max()
        max_y = circles_df['y'].max() + circles_df['radius'].max()
        print(f"Rect area: {(max_x - min_x)} * {(max_y - min_y)}")
        return (max_x - min_x) * (max_y - min_y)

    # Define a function to calculate the total area of all circles in a DataFrame
    def total_circles_area(circles_df):
        """
        Returns the total area of all circles in the given DataFrame.

        Parameters:
            circles_df (pandas.DataFrame): A DataFrame containing circle data.
                Each row of the DataFrame must have columns 'x', 'y', and 'radius',
                representing the x and y coordinates of the circle center and its radius.

        Returns:
            float: The sum of the areas of all circles in the DataFrame.
        """
        total_area = 0
        for i in range(len(circles_df)):
            total_area += math.pi * circles_df.iloc[i]['radius']**2
        return total_area
    
    print(f"Total overlap area: {total_overlap_area(circles_df):.2f} / total_circles_area : {total_circles_area(circles_df):.2f}")
    return total_overlap_area(circles_df) / total_circles_area(circles_df)

def weighted_average(metric, cluster_sizes):
    weights = [math.sqrt(size) for size, sim in zip(cluster_sizes, metric)]
    weighted_sum = sum(weight * sim for weight, sim in zip(weights, metric))
    sum_of_weights = sum(weights)
    weighted_avg= weighted_sum / sum_of_weights
    return weighted_avg