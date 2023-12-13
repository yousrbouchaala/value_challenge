from matplotlib import rcParams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from Model.Model import Model
from Data.Data_processing import Data


cv = CountVectorizer(ngram_range=(1, 2))
path_train = '/home/yousr/value_challenge/train_data.txt'
path_val = '/home/yousr/value_challenge/val_data.txt'
path_test = '/home/yousr/value_challenge/test_data.txt'

data_instance = Data(path_train=path_train, path_val=path_val , path_test=path_test)
data_instance.Process_data()
x, y = data_instance.x, data_instance.y

model_instance = Model(X=x, y=y)
trained_model = model_instance.train()


x_test, y_test = data_instance.Process_test_data(path_test=path_test)


predictions = trained_model.predict(x_test)

# Evaluate the model
rcParams['figure.figsize'] = 10, 5
acc_score = accuracy_score(y_test, predictions)
pre_score = precision_score(y_test, predictions)
rec_score = recall_score(y_test, predictions)
print('Accuracy_score:', acc_score)
print('Precision_score:', pre_score)
print('Recall_score:', rec_score)
print("-" * 50)
cr = classification_report(y_test, predictions)
print(cr)

model_instance.save_model('/home/yousr/value_challenge/model.pkl')
data_instance.save_vectorizer('/home/yousr/value_challenge/vectorizer.pkl')
