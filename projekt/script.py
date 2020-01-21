import pandas

pandas.options.mode.chained_assignment = None
import re

# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.similarities import MatrixSimilarity

import matplotlib.pyplot as plt

# Podanie słów kluczowych
keywordsInput = input("Słowa kluczowe oddzielone spacją: ")
hitsInput = input("Liczba filmów do wyświetlenia: ")
hitsInput = int(hitsInput)

dataset = pandas.read_csv('movies_metadata.csv', delimiter=',', low_memory=False, skip_blank_lines=True)

# Obliczenie ilości słów w opisach
dataset['word_count'] = dataset['overview'].apply(lambda x: len(str(x).split(" ")))

# Dodanie klucza keywords do datasetu
dataset['keywords'] = ' '

# Wyświetlenie tytułów, opisów wraz i ilości słów
# print(dataset[['original_title', 'overview', 'word_count']])

# Liczba filmów
# print('Liczba filmów: ', dataset['original_title'].count())

# Suma ilości słów
# print('Words: ', dataset['word_count'].sum())

# Szczegółowe opisanie rezultatów
# print(dataset.word_count.describe())

# Konwertowanie opisów filmów do stringów
dataset.overview = dataset.overview.astype(str)

# 20 najczęściej występujących słów
common_words = pandas.Series(' '.join(dataset['overview']).split()).value_counts()[:50]
# common_words.plot.bar()
# plt.show()
# print(common_words)

# Stopwords
stop_words = set(stopwords.words("english"))
stop_words.union(common_words)
# print(stop_words)

lem = WordNetLemmatizer()


def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist


import json

# 'Obróbka' opisów filmów
for i in range(0, dataset['original_title'].count()):
    text = dataset['overview'][i]

    if dataset['adult'][i]:
        text = text + ' children'
    else:
        text = text + ' adult'

    text = text + ' ' + dataset['original_title'][i]

    genres = json.loads('{"genres": ' + dataset['genres'][i].replace('\'', "\"") + '}')
    for g in genres['genres']:
        if g:
            text = text + ' ' + g['name']

    # print(dataset['original_title'][i])

    # production_companies = json.loads(
    #     '{"production_companies": ' + dataset['production_companies'][i].replace('\'', "\"") + '}')
    # for g in production_companies['production_companies']:
    #     if g:
    #         text = text + ' ' + g['name']

    # Usunięcie znaków interpunkcyjnych
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)

    # Zmiana na małe litery
    text = text.lower()

    # Usunięcie znaków specjalnych
    text = re.sub("(\\d|\\W)+", " ", text)

    # # Przekonwertowanie z napisu do listy
    text = text.split()

    # Lematyzacja
    text = [lem.lemmatize(word) for word in text if not word in stop_words]
    text = " ".join(text)

    # Usunięcie powtarzających się słów
    text = ' '.join(unique_list(text.split()))

    dataset['keywords'][i] = text

# 'Obróbka' słów kluczowych
keywordsInput = lem.lemmatize(keywordsInput.lower())
keywordsInput = keywordsInput.split()

dataset.reset_index(inplace=True)
movies = dataset[['original_title', 'keywords']]
keywords = movies['keywords'].tolist()
keywords = [word_tokenize(keyword.lower()) for keyword in keywords]

# Utworzenie słownika
dictionary = Dictionary(keywords)
corpus = [dictionary.doc2bow(doc) for doc in keywords]

# Model TFIDF
tfidf = TfidfModel(corpus)

# Macierz podobieństwa
sims = MatrixSimilarity(tfidf[corpus], num_features=len(dictionary))

# Konwersja listy wyrazów na listę postaci (token_id, token_count)
query_doc_bow = dictionary.doc2bow(keywordsInput)

# Konwersja podanego zestawu słów do modelu TFIDF
query_doc_tfidf = tfidf[query_doc_bow]

# Pobranie tablicy prawdopodobieństw pomiędzy słowami kluczowymi, a innymi filmami
similarity_array = sims[query_doc_tfidf]

# Konwersja do Serii
similarity_series = pandas.Series(similarity_array.tolist(), index=movies.original_title.values)

# Pobranie najbardziej pasujących rekordów
top_hits = similarity_series.sort_values(ascending=False)[:hitsInput]

# Wyświetlenie najbardziej pasujących rekordów
print("Najlepszych %s wyników dla słów %s:" % (hitsInput, keywordsInput))
for idx, (movie, score) in enumerate(zip(top_hits.index, top_hits)):
    print("%d '%s' z wynikiem podobieństwa %.3f" % (idx + 1, movie, score))
