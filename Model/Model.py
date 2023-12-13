import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def best_parametres(X, y):
    parameters = {

        'n_estimators': [500, 1000],
        'max_depth': [5, None],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [1, 5],
        'bootstrap': [True, False]
    }

    grid_search = GridSearchCV(RandomForestClassifier(), parameters, cv=5, return_train_score=True, n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_params_


class Model():
    def __init__(self, X, y):
        self.model = None
        self.X = X
        self.y = y

    def train(self):
        grid_search = best_parametres(self.X, self.y)

        rfc = RandomForestClassifier(
                                     max_depth=grid_search['max_depth'],
                                     n_estimators=grid_search['n_estimators'],
                                     min_samples_split=grid_search['min_samples_split'],
                                     min_samples_leaf=grid_search['min_samples_leaf'],
                                     bootstrap=grid_search['bootstrap'])
        rfc.fit(self.X, self.y)

        self.model = rfc
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)

    def save_model(self, filepath):
        joblib.dump(self.model, filepath)

