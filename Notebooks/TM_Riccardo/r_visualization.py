from typing import List
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.graph_objects as go
import re
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import euclidean
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import r_utils
import nltk
from collections import defaultdict
from ipywidgets import interact
from nltk.corpus import stopwords
from itertools import chain
nltk.download('stopwords')
from wordcloud import WordCloud


def visualize_documents_universal(
                        docs: List[str],
                        reduced_embeddings,
                        topics: List[int] = None,
                        sample: float = None,
                        hide_annotations: bool = False,
                        hide_document_hover: bool = False,
                        custom_labels: bool = False,
                        title: str = "<b>Documents and Topics</b>",
                        width: int = 1200,
                        height: int = 750,
                        topic_per_doc = None,
                        names = None):


    # Sample the data to optimize for visualization and dimensionality reduction
    if sample is None or sample > 1:
        sample = 1

    indices = []
    for topic in set(topic_per_doc):
        s = np.where(np.array(topic_per_doc) == topic)[0]
        size = len(s) if len(s) < 100 else int(len(s) * sample)
        indices.extend(np.random.choice(s, size=size, replace=False))
    indices = np.array(indices)

    df = pd.DataFrame({"topic": np.array(topic_per_doc)[indices]})
    df["doc"] = [docs[index] for index in indices]
    df["topic"] = [topic_per_doc[index] for index in indices]


    # Reduce input embeddings
    if reduced_embeddings is None:
        embeddings_2d = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    elif sample is not None and reduced_embeddings is not None:
        embeddings_2d = reduced_embeddings[indices]
    elif sample is None and reduced_embeddings is not None:
        embeddings_2d = reduced_embeddings

    unique_topics = set(topic_per_doc)
    if topics is None:
        topics = unique_topics

    # Combine data
    df["x"] = embeddings_2d[:, 0]
    df["y"] = embeddings_2d[:, 1]
    
    # Visualize
    fig = go.Figure()

    # Outliers and non-selected topics
    non_selected_topics = set(unique_topics).difference(topics)
    if len(non_selected_topics) == 0:
        non_selected_topics = [-1]

    selection = df.loc[df.topic.isin(non_selected_topics), :]
    selection["text"] = ""
    selection.loc[len(selection), :] = [None, None, selection.x.mean(), selection.y.mean(), "Other documents"]

    fig.add_trace(
        go.Scattergl(
            x=selection.x,
            y=selection.y,
            hovertext=selection.doc if not hide_document_hover else None,
            hoverinfo="text",
            mode='markers+text',
            name="other",
            showlegend=False,
            marker=dict(color='#CFD8DC', size=5, opacity=0.5)
        )
    )

    # Selected topics
    for name, topic in zip(names, unique_topics):
        if topic in topics and topic != -1:
            selection = df.loc[df.topic == topic, :]
            selection["text"] = ""

            if not hide_annotations:
                selection.loc[len(selection), :] = [None, None, selection.x.mean(), selection.y.mean(), name]

            fig.add_trace(
                go.Scattergl(
                    x=selection.x,
                    y=selection.y,
                    hovertext=selection.doc if not hide_document_hover else None,
                    hoverinfo="text",
                    text=selection.text,
                    mode='markers+text',
                    name=name,
                    textfont=dict(
                        size=12,
                    ),
                    marker=dict(size=5, opacity=0.5)
                )
            )

    # Add grid in a 'plus' shape
    x_range = (df.x.min() - abs((df.x.min()) * .15), df.x.max() + abs((df.x.max()) * .15))
    y_range = (df.y.min() - abs((df.y.min()) * .15), df.y.max() + abs((df.y.max()) * .15))
    fig.add_shape(type="line",
                  x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1],
                  line=dict(color="#CFD8DC", width=2))
    fig.add_shape(type="line",
                  x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2,
                  line=dict(color="#9E9E9E", width=2))
    fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)

    # Stylize layout
    fig.update_layout(
        template="simple_white",
        title={
            'text': f"{title}",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width,
        height=height
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig

def visualize_barchart_top2vec(model, n_words=5, top_n_topics = 10, reduced = False):
    data = np.ones((top_n_topics, n_words))
    words_matrix = []
    
    for i in range(1,top_n_topics+1):
        
        topic_words, word_scores, topic_nums = model.get_topics(i,reduced=reduced)
        data[i-1] = word_scores[i-1][0:n_words]
        # word scores is the cosine similarity between word and centroid cluster 
        words_matrix.append(topic_words[i-1][0:n_words]) 
    
    n_cols = 4
    n_rows = (top_n_topics + n_cols - 1) // n_cols
    # Create top_n_topics horizontal bar plots, 4 per line
    for i in range(0, top_n_topics, 4):
        fig, axs = plt.subplots(1, 4, figsize=(12, 3))
        for j in range(n_cols):
            
            idx = i + j
            if idx >= top_n_topics:
                break
            # Sort the data for the i+j-th bar plot
            sorted_data = np.sort(data[i+j])[::-1]

            y_pos = words_matrix[i+j]

            # Create a horizontal bar plot
            axs[j].barh(y_pos, sorted_data)
            axs[j].set_title('Topic {}'.format(i+j+1))
            axs[j].set_xlabel('Value')
            axs[j].set_ylabel('Index')
            axs[j].set_yticks(y_pos)
            axs[j].invert_yaxis()
        plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
        plt.show()   
        
        
        
        
        
        

def get_id2word_corpus(docs):
    """
    This function takes in a list of documents and preprocesses the text, removes stopwords and returns
    the id2word dictionary and corpus using gensim's Dictionary and Corpus functions.
    
    Args:
    - docs: a list of documents
    
    Returns:
    - id2word: a gensim Dictionary object
    - corpus_tf: a corpus in bag-of-words format, as a list of lists (document-wise)
    """
    
    
    df_tmp = pd.DataFrame({"docs": docs})
    df_tmp["post_preprocessed"] = df_tmp["docs"].apply(r_utils.preprocess_text).str.lower()

    print('lemming...')

    lemmatizer = WordNetLemmatizer()
    df_tmp["post_final"] = df_tmp["post_preprocessed"].apply(lambda post: r_utils.lemmatize_words(post, lemmatizer))

    print('remove stopwords...')

    swords = set(stopwords.words('english'))

    df_tmp["post_final"] = df_tmp["post_final"].apply(lambda post: r_utils.remove_stopwords(post, swords))
    
    posts = [x.split(' ') for x in df_tmp["post_final"]]
    
    id2word = corpora.Dictionary(posts)
    corpus_tf = [id2word.doc2bow(text) for text in posts]
    
    return id2word, corpus_tf


def compute_words_frequency_total(id2word, corpus):
    """
    Compute the total frequency of each term in the entire corpus.

    Args:
    - id2word (corpora.Dictionary): A dictionary that maps each term to a unique ID.
    - corpus (list of lists): A list of documents, where each document is represented as a list of (term_id, frequency) tuples.

    Returns:
    - term_freqs (dict): A dictionary that maps each term to its total frequency in the corpus, sorted in descending order.
    """
    
    # Create a dictionary to store the term frequencies across all documents
    term_freqs = {}

    # Loop over all documents in the corpus
    for doc in corpus:
        # Loop over all terms in the document
        for term_id, freq in doc:
            # Get the term from its ID
            term = id2word[term_id]
            # Update the term frequency in the dictionary
            if term not in term_freqs:
                term_freqs[term] = freq
            else:
                term_freqs[term] += freq
                
    # total frequency in the corpus
    term_freqs = dict(sorted(term_freqs.items(), key=lambda x: x[1], reverse=True))
    return term_freqs

def compute_words_frequency_per_topic(labels, id2word, corpus):
    """    
    Computes the frequency of each word within each topic.
    
    Args:
    - labels (list): A list of topic labels, where the i-th label corresponds to the i-th document in the corpus.
    - id2word (gensim.corpora.dictionary.Dictionary): A gensim dictionary object representing the mapping between words and 
    their integer ids.
     - corpus (list of list of tuples): A list of document-term frequency lists, where each list contains tuples of 
     (word_id, frequency) for the words in the document.
        
    Returns:
        A defaultdict containing the word frequencies for each topic. The keys of the outer dictionary are the topic labels, and the 
        values are dictionaries that map each word to its frequency within that topic.
    """
    
    # Create a dictionary to hold the word frequencies for each topic
    label_word_freqs = defaultdict(lambda: defaultdict(int))

    for i, doc in enumerate(corpus):
        label = labels[i]
        for term_id, freq in doc: 
            term = id2word[term_id]
            label_word_freqs[label][term] += freq
            
    return label_word_freqs

def extract_set(df):
    """
    Extracts a set of unique words from the "top_10_words" column of a DataFrame.
    
    Args:
    - df: a DataFrame containing a column of top 10 words
    
    Returns:
    - a set containing all unique words in the top 10 lists across all rows
    """
    wl = []
    for i in range(df.shape[0]):
        wl.append(df.iloc[i].top_10_words.split())
    return set(chain.from_iterable(wl))

def keep_important_words(label_word_freqs, words_set):
    """
    Filters out non-important words from the word frequency dictionary for each topic.
    
    Args:
    - label_word_freqs: a dictionary of word frequencies per topic
    - words_set: a set of important words
    
    Returns:
    - the same dictionary with all non-important words removed
    """
    for topic, word_freq in label_word_freqs.items():
        for word in list(word_freq.keys()):
            if word not in words_set:
                del word_freq[word]
    return label_word_freqs

def sort_result(result):
    """
    Sorts the dictionary of top 10 words per topic.
    
    Args:
    - result: a dictionary of top 10 words per topic
    
    Returns:
    - the same dictionary with the following sorting:
        - Each value dictionary is sorted by the ratio of the first two numbers in its value string (split by '-'), in descending order.
        - Each value dictionary is then truncated to contain only the top 10 words.
        - The key-value pairs in the entire dictionary are sorted by the third number in the value string, in descending order.
        - Finally, the dictionary is sorted by the keys in ascending order.
    """
    for key, value in result.items():
        sorted_values = sorted(value.items(), key=lambda x: int(x[1].split('-')[0]) / int(x[1].split('-')[1]), reverse=True)
        top10_values = sorted_values[:10]
        top10_dict = dict(top10_values)
        result[key] = top10_dict
        
    sorted_dict = {}
    for key, value in result.items():
        sorted_dict[key] = {k: v for k, v in sorted(value.items(), key=lambda item: int(item[1].split('-')[1]), reverse=True)}
    
    # order for ascending key
    sorted_dict = dict(sorted(sorted_dict.items()))
    return sorted_dict

def interactive_plot(data):
    def plot_bar(key):
        x = []
        y1 = []
        y2 = []
        for i, (word, values) in enumerate(data[key].items()):
            x.append(word)
            val1, val2 = values.split('-')
#             y1.append(int(val1))
#             y2.append(int(val2))
            y1.append(int(val1)/int(val2))
            y2.append(1)
        plt.figure(figsize=(10, 5))
        plt.barh(x, y1, color='blue', alpha=0.5, label='frequency per topic')
        plt.barh(x, y2, color='gray', alpha=0.5, label='total frequency')
        plt.xlabel('frequency per topic / total frequency')
        plt.ylabel('Words')
        plt.title(f'Bar plot for {key}')
        plt.ylim(-1, len(x))
        plt.xlim(0, max(y2)*1.1)
        plt.legend()
        plt.show()

    interact(plot_bar, key=list(data.keys()))
    
    
    
def plot_interactive_frequence_top15words(docs, labels, df_words):
    # calculate the id2word and the corpus
    id2word, corpus = get_id2word_corpus(docs)

    # compute words frequency across all documents
    terms_freqs = compute_words_frequency_total(id2word, corpus)
    
    # compute words frequency inside each topic
    label_word_freqs = compute_words_frequency_per_topic(labels, id2word, corpus)
    
    # select the most 15 frequent words for each topic
#     label_word_freqs_top15 = select_15_most_frequent_words_per_topic(label_word_freqs)
    words_set = extract_set(df_words)
    
    label_word_freqs_important_words = keep_important_words(label_word_freqs, words_set)
    
    # join between the 2 dict
    result = {k1: {k2: str(v1) + "-" + str(terms_freqs[k2]) for k2, v1 in v1.items()} for k1, v1 in label_word_freqs_important_words.items()}
    
    # sort by highest values and key (just for better visualization)
    sorted_result = sort_result(result)
    
    # interactive plot
    interactive_plot(sorted_result)
    
    
    
def plt_cluster_frequency(frequencies, model_names):
    # Get the number of clusters
    n_clusters = len(frequencies[0])

    # Plot the bar plots for each model next to each other
    fig, ax = plt.subplots(figsize=(16,5))
    width = 1.0 / (len(model_names) + 1)
    for i, model_name in enumerate(model_names):
        x = np.arange(n_clusters) + i*width
        y = frequencies[i]
        ax.bar(x, y, width=width, label=model_name)

    ax.set_xticks(np.arange(n_clusters))
    ax.set_xticklabels(np.arange(n_clusters))
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of documents")
    ax.set_ylim(0, max(max(y) for y in frequencies) + 500)
    ax.grid(True)
    ax.set_title("Number of documents in each cluster")
    plt.title('(topics plotted by ascending values -> topics do not match between models)', fontsize=10)
    ax.legend()

    plt.show()

def create_wordcloud(topic_model, topic):
    text = {word: value for word, value in topic_model.get_topic(topic)}
    wc = WordCloud(background_color="white", max_words=1000)
    wc.generate_from_frequencies(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    
    
def compute_centroid_radius(reduced_embeddings, labels):
    """
    Computes the centroid and radius of each cluster using the UMAP algorithm for dimensionality reduction and the 
    Euclidean distance between each point and its centroid. 

    Args:
    - embeddings: A numpy array of shape (n_samples, n_features) representing the embeddings of the data.
    - labels: A numpy array of shape (n_samples,) representing the cluster labels of each sample.

    Returns:
    - A pandas DataFrame containing the coordinates of the centroid (columns 'x' and 'y'), the label of the cluster
    (column 'label') and the radius of the cluster (column 'radius').

    """
#     reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    
    df_embeddings = pd.DataFrame({"x":reduced_embeddings[:,0], "y": reduced_embeddings[:,1], "label": labels})
    
    df_centroids = df_embeddings.groupby(by=["label"]).mean()

    df_tmp = pd.merge(df_embeddings,df_centroids, on="label")
        
    distances = []
    for index, row in df_tmp.iterrows():
        point1 = [row['x_x'], row['y_x']]
        point2 = [row['x_y'], row['y_y']]
        distance = euclidean(point1, point2)
        distances.append(distance)

    # create a new column in the DataFrame to store the Euclidean distance values
    df_tmp['euclidean_distance'] = distances

    df_eucl_dist = df_tmp.groupby(by=["label"])["euclidean_distance"].mean()
    df_centroids["radius"] = df_eucl_dist.values
    return df_centroids

def plot_centroid_radius_frequency(df,title):

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(df['x'], df['y'])
    # add circles to the plot with radius and color based on frequency
    for index, row in df.iterrows():
        circle = plt.Circle((row['x'], row['y']), row['radius'], alpha=0.5, 
                             color=plt.cm.YlGnBu((row['frequence'] - df['frequence'].min()) / (df['frequence'].max() - df['frequence'].min())),
                             label=row['frequence'])
        ax.add_artist(circle)

    # adjust axis limits to fit the circles
    ax.set_xlim([df['x'].min() - df['radius'].max(), df['x'].max() + df['radius'].max()])
    ax.set_ylim([df['y'].min() - df['radius'].max(), df['y'].max() + df['radius'].max()])

    # add legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title='Frequency', loc='lower right')
    plt.title(title)
    # show the plot
    plt.show()
    
    
    
    
import r_utils
def get_id2word_corpus_2(docs):
    """
    This function takes in a list of documents and preprocesses the text, removes stopwords and returns
    the id2word dictionary and corpus using gensim's Dictionary and Corpus functions.
    
    Args:
    - docs: a list of documents
    
    Returns:
    - id2word: a gensim Dictionary object
    - corpus_tf: a corpus in bag-of-words format, as a list of lists (document-wise)
    """
    
    
    df_tmp = pd.DataFrame({"docs": docs})
    df_tmp["post_preprocessed"] = df_tmp["docs"].apply(r_utils.preprocess_text).str.lower()

#     print('lemming...')

#     lemmatizer = WordNetLemmatizer()
#     df_tmp["post_final"] = df_tmp["post_preprocessed"].apply(lambda post: r_utils.lemmatize_words(post, lemmatizer))

#     print('remove stopwords...')

    swords = set(stopwords.words('english'))

#     df_tmp["post_final"] = df_tmp["post_final"].apply(lambda post: r_utils.remove_stopwords(post, swords))
#     df_tmp["post_final"] = df_tmp["post_preprocessed"].apply(lambda post: r_utils.remove_stopwords(post, swords))
    
    posts = [x.split(' ') for x in df_tmp["post_preprocessed"]]
    
    id2word = corpora.Dictionary(posts)
    corpus_tf = [id2word.doc2bow(text) for text in posts]
    
    return id2word, corpus_tf

def get_corpus_topic(df_topics, topic):
    docs_k = df_topics[ df_topics["labels"] == topic]
    corpus_k = docs_k.corpus
    return corpus_k.tolist()

def count_docs_with_word(corpus, word_id):
    count = 0
    for doc in corpus:
        for word in doc:
            if word[0] == word_id:
                count += 1
                break
    return count

def compute_score(df_topics, df_words, word2id, id2word, corpus, bert_flag=True):
    topic = 0
   
    topic_m_values = {}
    for el in df_words.top_10_words.tolist():
        corpus_k = get_corpus_topic(df_topics, topic=topic)
        word_m_values = {}
        for word in el.split():

            if word in id2word.token2id:
                id_word = word2id[word]

                number_docs_word_in_cluster_k = count_docs_with_word(corpus_k, id_word)
                number_docs_in_cluster_k = len(corpus_k)

                number_docs_word = count_docs_with_word(corpus, id_word)
                number_docs = len(corpus)
                m = np.log (  (number_docs_word_in_cluster_k / number_docs_in_cluster_k) * (number_docs / number_docs_word ) )
                if m < -100:
                    m = 0
#                 print(f"Word: {word}, metric: {m:.2f}, ({number_docs_word_in_cluster_k:.2f} / {number_docs_in_cluster_k:.2f}) / ({number_docs_word:.2f} / {number_docs:.2f}) = ({number_docs_word_in_cluster_k / number_docs_in_cluster_k:.2f}) / ({number_docs_word/ number_docs:.2f})")
            else:
                m = 0 
            word_m_values[word] = m
        topic_m_values[topic] = word_m_values
        topic += 1
    return topic_m_values



def plot_interactive(data,nr_topics):
    # Create the figure object
    fig = go.Figure()

    # Iterate over each value of k
    for k in range(nr_topics):
        # Get the words and scores for the current k
        words, scores = zip(*data[k].items())
        # Create the horizontal bar plot for the current k
        fig.add_trace(go.Bar(
            x=scores,
            y=words,
            orientation="h",
            name=f"k={k}"
        ))

    # Add layout information
    fig.update_layout(
        title="Word scores for different k values",
        xaxis_title="Score",
        yaxis_title="Word",
        barmode="stack",
        height=800
    )

    # Show the figure
    fig.show()


def get_tfidf_top10words_normalized(docs, labels, df_words, bert_flag=True):
    # calculate the id2word and the corpus
    id2word, corpus = get_id2word_corpus_2(docs)
    
    # data structures needed 
    df_tmp = pd.DataFrame({"docs":docs,"labels":labels, "corpus":corpus})
    word2id = dict((v, k) for k, v in id2word.items())
    
    topic_values = compute_score(df_tmp, df_words, word2id, id2word, corpus, bert_flag)
    nr_topics = len(set(labels))
    if bert_flag:
        nr_topics = len(set(labels))-1
    
    return topic_values, nr_topics

def plot_tfidf_values(topic_values, nr_topics):
    plot_interactive(topic_values,nr_topics)
