import pandas as pd
import numpy as np
import re


def format_name(df):
    def select_3_words(data):
        return data.split()[:3]
    
    def generate_topic_name(topic_number, words):
        return f"{topic_number}_{words[0]}_{words[1]}_{words[2]}"

    names = list(df.top_10_words)
    
    list_names = []
    for name in names:
        list_names.append(select_3_words(name))
        
    list_names1 = []
    i = 0
    for el in list_names:
        list_names1.append(generate_topic_name(i,el))
        i += 1
    return list_names1

def get_words_list_top2vec(model, reduced=True):
    def get_top_words_top2vec(topic):
        topic_words, word_scores, topic_nums = model.get_topics(topic,reduced=reduced)
        top_50_words_list = topic_words[topic-1]
        return top_50_words_list[0:10].tolist()
    id_topic = []
    words_list = []
    for i in range(1, model.get_num_topics(reduced=reduced)+1):
        id_topic.append(i)
        words_list.append(get_top_words_top2vec(i))
        
    return words_list

def get_words_list_bertopic(model, outliers = True):
    if outliers:
        words_list = [[words for words, _ in model.get_topic(topic)[0:10]] 
                       for topic in range(len(set(model.topics_))-1)]
    else:
        words_list = [[words for words, _ in model.get_topic(topic)[0:10]] 
                       for topic in range(len(set(model.topics_)))]
        
    return words_list

def get_words_list_lda(model, num_topics):
    words_list = []
    for i in range(num_topics):
        words_prob = model.show_topic(i)
        words = [t[0] for t in words_prob]
        words_list.append(words)
    return words_list


def get_df_top10words_BERTopic(model, nr_topics = 10, outliers=True):
    top10words = []
    id_topic = []
    if outliers == True:
        end = nr_topics-1
    else:
        end = nr_topics
    for i in range(0,end):
        id_topic.append(i)
        topic_dict = model.topic_representations_[i]
        x = [t[0] for t in topic_dict]
        top10words.append(' '.join(x[0:10]))
        
    df = pd.DataFrame({"id_topic":id_topic, "top_10_words": top10words})

    return df

def get_df_top10words_top2vec(model, nr_topics = 10, reduced = False):
    
    def get_top10words_top2vec(model,topic,reduced = False):
        
        topic_words, word_scores, topic_nums = model.get_topics(topic,reduced=reduced)
        if topic == 0:
            top_50_words_list = ""
        else:
            top_50_words_list = topic_words[topic-1]
        return ' '.join(top_50_words_list[0:10])
    
    top10words = []
    id_topic = []
    for i in range(1, model.get_num_topics(reduced=reduced)+1):
        id_topic.append(i)
        top10words.append(get_top10words_top2vec(model,i,reduced=reduced))
    df = pd.DataFrame({"id_topic":np.array(id_topic)-1, "top_10_words": top10words})
    return df






def remove_urls(text):
    " removes urls"
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)
    
def remove_html(text):
    " removes html tags"
    html_pattern = re.compile('')
    return html_pattern.sub(r'', text)

def remove_emails(text):
    email_pattern = re.compile('\S*@\S*\s?')
    return email_pattern.sub(r'', text)

def remove_new_line(text):
    return re.sub('\s+', ' ', text)

def remove_non_alpha(text):
    return re.sub("[^A-Za-z]+", ' ', str(text))

def preprocess_text(text):
    t = remove_urls(text)
    t = remove_html(t)
    t = remove_emails(t)
    t = remove_new_line(t)
    t = remove_non_alpha(t)
    return t

def lemmatize_words(text, lemmatizer):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stopwords(text, stopwords):
    return " ".join([word for word in str(text).split() if word not in stopwords])


