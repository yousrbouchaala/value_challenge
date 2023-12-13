import joblib
import pandas as pd
import re
import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer


def text_transformation(df_col):
    lm = WordNetLemmatizer()
    corpus = []

    for item in df_col:
        new_item = re.sub('[^a-zA-Z]', ' ', str(item))
        new_item = new_item.lower()
        new_item = new_item.split()
        new_item = [lm.lemmatize(word) for word in new_item if word not in set(stopwords.words('english'))]

        corpus.append(' '.join(str(x) for x in new_item))
    return corpus


def label_encoder(df):
    df.replace(to_replace=" satisfait", value=1, inplace=True)
    df.replace(to_replace=" non_satisfait", value=0, inplace=True)


class Data:

    def __init__(self, path_train =None, path_val=None, path_test = None, x=None, y=None  ):
        self.x = x
        self.y = y
        self.train = path_train
        self.val = path_val
        self.path_test = path_test
        self.cv = None

    def Process_data(self):
        df_train = pd.read_csv(self.train, delimiter=';', names=['text', 'label'])
        if self.val:

            df_val = pd.read_csv(self.val, delimiter=';', names=['text', 'label'])

            df = pd.concat([df_train, df_val])
        else:
            df = df_train
        # df.reset_index(inplace=True, drop=True)
        label_encoder(df)
        corpus = text_transformation(df['text'])

        self.cv = CountVectorizer(ngram_range=(1, 2))

        train_vector = self.cv.fit_transform(corpus)
        self.x = train_vector
        self.y = df.label.astype(int)

    def Process_test_data(self, path_test):
        df_test = pd.read_csv(path_test, delimiter=';', names=['text', 'label'])
        label_encoder(df_test)

        corpus_test = text_transformation(df_test['text'])
        test_vector = self.cv.transform(corpus_test)

        return test_vector, df_test.label.astype(int)

    def save_vectorizer(self, filepath):
        joblib.dump(self.cv, filepath)
