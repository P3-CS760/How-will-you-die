import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB
RANDOM_STATE = 1234

PATH_ROOT = Path(os.getcwd())
PATH = os.path.join(PATH_ROOT.parent, 'Model', 'CleanDf.csv')
data = pd.read_csv(PATH)
data = data.drop(labels="Id", axis=1)
for i, row in data.iterrows():
    if isinstance(row["Age of death"], str):
        data.drop(data.index(i))
data = data.dropna()



X, y = data[["PrePro_ShortDesc", "Birth year", "Death year"]], data["Age of death"] // 10

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)\

training_df = pd.DataFrame(X_train)
training_df.columns = ["PrePro_ShortDesc", "Birth year", "Death year"]

testing_df = pd.DataFrame(X_test)
testing_df.columns = ["PrePro_ShortDesc", "Birth year", "Death year"]

X_train_desc, X_train_birth, X_train_death = training_df["PrePro_ShortDesc"].to_numpy(), training_df["Birth year"].to_numpy(), training_df["Death year"].to_numpy()
X_test_desc, X_test_birth, X_test_death = testing_df["PrePro_ShortDesc"].to_numpy(), testing_df["Birth year"].to_numpy(), testing_df["Death year"].to_numpy()

count_vec = CountVectorizer(stop_words='english')
X_train_vectors = count_vec.fit_transform(X_train_desc).toarray()
X_test_vectors = count_vec.transform(X_test_desc).toarray()

X_train_birth = X_train_birth.reshape((X_train_birth.size, 1))
X_train_death = X_train_death.reshape((X_train_death.size, 1))
X_test_birth = X_test_birth.reshape((X_test_birth.size, 1))
X_test_death = X_test_death.reshape((X_test_death.size, 1))

X_train = np.hstack((X_train_vectors, X_train_birth, X_train_death))
X_test = np.hstack((X_test_vectors, X_test_death, X_test_death))

complement_classifier = ComplementNB()
multinomial_classifier = MultinomialNB()
gaussian_classifier = GaussianNB()

complement_parameters = {'alpha': (1, 2, 3, 4, 5), 'fit_prior': (True, False), 'norm': (True, False)}
multinomial_parameters = {'alpha': (1, 2, 3, 4, 5), 'fit_prior': (True, False)}

complement_clf = GridSearchCV(complement_classifier, complement_parameters, refit=True, cv=5)
complement_clf.fit(X_train, y_train)

multinomial_clf = GridSearchCV(multinomial_classifier, multinomial_parameters, refit=True, cv=5)
multinomial_clf.fit(X_train, y_train)

gaussian_classifier.fit(X_train, y_train)

print(complement_clf.score(X_test, y_test))
print(multinomial_clf.score(X_test, y_test))
print(gaussian_classifier.score(X_test, y_test))




