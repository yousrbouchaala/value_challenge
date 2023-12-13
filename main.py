import re
import joblib
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords


def preprocess_input(user_input, vectorizer):
    lm = WordNetLemmatizer()
    user_input_processed = re.sub('[^a-zA-Z]', ' ', str(user_input))
    user_input_processed = user_input_processed.lower()
    user_input_processed = user_input_processed.split()
    user_input_processed = [lm.lemmatize(word) for word in user_input_processed if
                            word not in set(stopwords.words('english'))]

    user_input_vectorized = vectorizer.transform([' '.join(user_input_processed)])

    return user_input_vectorized


loaded_model = joblib.load('/home/yousr/value_challenge/model.pkl')

loaded_vectorizer = joblib.load('/home/yousr/value_challenge/vectorizer.pkl')

user_input = input("Enter Your Review about Value: ")
processed_input = preprocess_input(user_input, loaded_vectorizer)
prediction = loaded_model.predict(processed_input)

if prediction == 0:
    print("Non satisfait :( ")

else:
    print("Satisfait :) ")
