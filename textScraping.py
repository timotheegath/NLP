import requests
from bs4 import BeautifulSoup
import pandas as pd
import spacy
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import WordNetLemmatizer
import unicodedata
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from extDependencies.contractions import CONTRACTION_MAP

seed_urls = ['https://inshorts.com/en/read/technology',
             'https://inshorts.com/en/read/sports',
             'https://inshorts.com/en/read/world']


def build_dataset(seed_urls):
    news_data = []
    for url in seed_urls:
        news_category = url.split('/')[-1]
        data = requests.get(url)
        soup = BeautifulSoup(data.content, 'html.parser')

        news_articles = [{'news_headline': headline.find('span',
                                                         attrs={"itemprop": "headline"}).string,
                          'news_article': article.find('div',
                                                       attrs={"itemprop": "articleBody"}).string,
                          'news_category': news_category}

                         for headline, article in
                         zip(soup.find_all('div',
                                           class_=["news-card-title news-right-box"]),
                             soup.find_all('div',
                                           class_=["news-card-content news-right-box"]))
                         ]
        news_data.extend(news_articles)
    df = pd.DataFrame(news_data)
    df = df[['news_headline', 'news_article', 'news_category']]
    return df


def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())

        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text


def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text


def lemmatize_text(text):
    lematizer = WordNetLemmatizer()
    toktok = ToktokTokenizer()

    text = ' '.join([lematizer.lemmatize(word) for word in toktok.tokenize(text)])
    return text


sentence = 'Making complicated sentences could lose all meaning using the nltk porter stemmer'
print(simple_stemmer(sentence))
print(lemmatize_text(sentence))
news_df = build_dataset(seed_urls)


nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)
#nlp_vec = spacy.load('en_vecs', parse = True, tag=True, #entity=True)
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')




