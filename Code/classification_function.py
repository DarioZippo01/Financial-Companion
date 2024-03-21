import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import learning_curve, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from skopt import BayesSearchCV
from matplotlib import pyplot as plt


def plot_learning_curves(model, X, y, model_name):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=10, scoring='accuracy')

    # Calcola gli errori su addestramento e test
    train_errors = 1 - train_scores
    test_errors = 1 - test_scores
    # Calcola la deviazione standard e la varianza degli errori su addestramento e test
    train_errors_std = np.std(train_errors, axis=1)
    test_errors_std = np.std(test_errors, axis=1)
    train_errors_var = np.var(train_errors, axis=1)
    test_errors_var = np.var(test_errors, axis=1)

    # Stampa i valori numerici della deviazione standard e della varianza
    print(
        f"\033[95m{model_name} - Train Error Std: {train_errors_std[-1]}, Test Error Std: {test_errors_std[-1]}, Train Error Var: {train_errors_var[-1]}, Test Error Var: {test_errors_var[-1]}\033[0m")

    # Calcola gli errori medi su addestramento e test
    mean_train_errors = 1 - np.mean(train_scores, axis=1)
    mean_test_errors = 1 - np.mean(test_scores, axis=1)

    #Visualizza la curva di apprendimento
    plt.figure(figsize=(16, 10))
    plt.plot(train_sizes, mean_train_errors, label='Errore di training', color='green')
    plt.plot(train_sizes, mean_test_errors, label='Errore di testing', color='red')
    plt.title(f'Curva di apprendimento per {model_name}')
    plt.xlabel('Dimensione del training set')
    plt.ylabel('Errore')
    plt.legend()
    plt.show()


def best_params_knn(training,target):
    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size=0.3, random_state=42)
    knn = KNeighborsClassifier()
    
    # Ricerca dei best params
    hyperparameters_knn = {
        "n_neighbors": (5, 7, 9, 13),
        "weights": ("uniform", "distance"),     
        "metric": ("minkowski", "manhattan")}
    
    #Grid Search
    grid_search_knn = GridSearchCV(
        estimator = knn,
        param_grid = hyperparameters_knn,
        scoring = "accuracy",
        n_jobs = -1,
        cv = 5,
    )
    #Bayesian Search
    
    bayes_search_knn = BayesSearchCV(estimator=knn,
                    search_spaces = hyperparameters_knn,
                    cv=5,
                    n_jobs=8,
                    scoring='accuracy',
                    random_state=123)
    
    #Random Search
    
    random_search_knn = RandomizedSearchCV(estimator = knn,
                         param_distributions = hyperparameters_knn,
                         cv=5,
                         n_jobs=8,
                         scoring='accuracy',
                         random_state=123)
    np.int = int
    knn_grid = grid_search_knn.fit(x_train, y_train)
    knn_bayes = bayes_search_knn.fit(x_train, y_train)
    knn_random = random_search_knn.fit(x_train, y_train)
    
def knn_classification(training, target):
    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size=0.3, random_state=42)
    knn = KNeighborsClassifier(metric="minkowski", n_neighbors=1, weights="uniform")
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)

    print("Accuracy knn test set:", metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))
    
    y_pred = knn.predict(x_train)
    print("Accuracy knn training set:", metrics.accuracy_score(y_train, y_pred))
    print(metrics.classification_report(y_train, y_pred))
    
    plot_learning_curves(knn, training.to_numpy(), target.to_numpy(), "KNN")
    
    return knn

def best_params_multinomial_bayes(training, target):
    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size=0.3, random_state=42)
    multinomial_bayes = MultinomialNB()
    
    # Ricerca dei best params
    hyperparameters_multinomial_bayes = {
        'MultinomialNB__alpha': (100, 300, 400, 500, 700, 900)}
    p = Pipeline([('Normalizing',MinMaxScaler()),('MultinomialNB',MultinomialNB(fit_prior=False))])
    grid_search_multinomial_bayes = GridSearchCV(
        estimator=p,
        param_grid=hyperparameters_multinomial_bayes,
        cv=10,  # use any cross validation technique
        verbose=1,
        scoring='accuracy'
    )

    #Bayesian Search
    
    bayes_search_multinomial_bayes = BayesSearchCV(estimator=p,
                    search_spaces = hyperparameters_multinomial_bayes,
                    cv=5,
                    n_jobs=8,
                    scoring='accuracy',
                    random_state=123)
    
    #Random Search
    
    '''random_search_multinomial_bayes = RandomizedSearchCV(estimator = p,
                         param_distributions = hyperparameters_multinomial_bayes,
                         cv=5,
                         n_jobs=8,
                         scoring='accuracy',
                         random_state=123)'''
    np.int = int
    multinomial_bayes_grid = grid_search_multinomial_bayes.fit(x_train, y_train)
    multinomial_bayes_bayes = bayes_search_multinomial_bayes.fit(x_train, y_train)
    #multinomial_bayes_random = random_search_multinomial_bayes.fit(x_train, y_train)
    
    print("Best params Grid", grid_search_multinomial_bayes.best_params_)
    print("Best params Bayes", bayes_search_multinomial_bayes.best_params_)
    #print("Best params Random", random_search_multinomial_bayes.best_params_)
    
def multinomial_bayes_classification(training, target):

    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size=0.3, random_state=42)
    multinomial_bayes = MultinomialNB()
    p = Pipeline([('Normalizing',MinMaxScaler()),('MultinomialNB',MultinomialNB(fit_prior=False, alpha=100))])
    p.fit(x_train, y_train)
    y_pred_multinomial_bayes = p.predict(x_test)
    test_acc = metrics.accuracy_score(y_test, y_pred_multinomial_bayes)
    print("Accuracy multinomial bayes test set:", test_acc)

    print(metrics.classification_report(y_test, y_pred_multinomial_bayes))
    
    y_pred = p.predict(x_train)
    train_acc = metrics.accuracy_score(y_train, y_pred)
    print("Accuracy multinomial bayes training set:", train_acc)
    print(metrics.classification_report(y_train, y_pred))
    
    plot_learning_curves(p, training.to_numpy(), target.to_numpy(), "Multinomial Bayes")
    return p

def best_params_rf(training, target):
    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size=0.3, random_state=42)
    rf = RandomForestClassifier()
    
    # Ricerca dei best params
    hyperparameters_rf = {
        'n_estimators': (100, 500, 800, 1200),
        'max_depth': (15, 25, 35),
        'min_samples_split': (2, 5, 10),
        'min_samples_leaf': (1, 2, 4)
    }

    '''grid_search_rf = GridSearchCV(rf, hyperparameters_rf, cv=10, verbose=1,
                         scoring='accuracy')'''
        #Bayesian Search
    
    bayes_search_rf = BayesSearchCV(estimator=rf,
                    search_spaces = hyperparameters_rf,
                    cv=10,
                    n_jobs=8,
                    scoring='accuracy',
                    random_state=123)
    
    #Random Search
    
    random_search_rf = RandomizedSearchCV(estimator = rf,
                         param_distributions = hyperparameters_rf,
                         cv=10,
                         n_jobs=8,
                         scoring='accuracy',
                         random_state=123)
    np.int = int
    #rf_grid = grid_search_rf.fit(x_train, y_train)
    rf_bayes = bayes_search_rf.fit(x_train, y_train)
    rf_random = random_search_rf.fit(x_train, y_train)
    
    #print("Best params Grid", grid_search_rf.best_params_)
    print("Best params Bayes", bayes_search_rf.best_params_)
    print("Best params Random", random_search_rf.best_params_)
    
    
    
    
def random_forest_classification(training, target):
    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size=0.3, random_state=42)
    rf = RandomForestClassifier(max_depth=24, min_samples_leaf=2, min_samples_split=4, n_estimators=1300)
    rf.fit(x_train.values, y_train)
    y_pred_rf = rf.predict(x_test.values)
    print("Accuracy rf test set:", metrics.accuracy_score(y_test, y_pred_rf))

    print(metrics.classification_report(y_test, y_pred_rf))
    
    y_pred = rf.predict(x_train)
    print("Accuracy rf training set:", metrics.accuracy_score(y_train, y_pred))
    print(metrics.classification_report(y_train, y_pred))
    
    plot_learning_curves(rf, training.to_numpy(), target.to_numpy(), "RF")
    return rf

def random_forest_classification_standalone(training, target):
    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size=0.3, random_state=42)
    rf = RandomForestClassifier(max_depth=24, min_samples_leaf=2, min_samples_split=4, n_estimators=1300)
    rf.fit(x_train.values, y_train)
    return rf

def best_params_clf(training, target):
    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size=0.3, random_state=42)
    clf = LogisticRegression(solver='newton-cg')
    # Ricerca dei best params
    hyperparameters_clf = {
        'C' : (100, 10, 1.0, 0.1, 0.01),
        'max_iter' : (25, 50, 100)
    }

    grid_search_clf = GridSearchCV(clf, hyperparameters_clf, cv=10, verbose=1,
                         scoring='accuracy')
        #Bayesian Search
    
    bayes_search_clf = BayesSearchCV(estimator=clf,
                    search_spaces = hyperparameters_clf,
                    cv=10,
                    n_jobs=8,
                    scoring='accuracy',
                    random_state=123)
    
    #Random Search
    
    '''random_search_clf = RandomizedSearchCV(estimator = clf,
                         param_distributions = hyperparameters_clf,
                         cv=10,
                         n_jobs=8,
                         scoring='accuracy',
                         random_state=123)'''
    np.int = int
    clf_grid = grid_search_clf.fit(x_train, y_train)
    clf_bayes = bayes_search_clf.fit(x_train, y_train)
    #clf_random = random_search_clf.fit(x_train, y_train)
    
    print("Best params Grid", grid_search_clf.best_params_)
    print("Best params Bayes", bayes_search_clf.best_params_)
    #print("Best params Random", random_search_clf.best_params_)


def logistic_regression_classification(training, target):
    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size=0.3, random_state=42, shuffle=True)
    clf = LogisticRegression(multi_class='auto', random_state=42, penalty='l2', solver='newton-cg', C=100, max_iter=100) #10 perché ho un dataset abbastanza grande
    clf.fit(x_train, y_train)
    y_pred_rf = clf.predict(x_test)
    print("Accuracy clf test set:", metrics.accuracy_score(y_test, y_pred_rf))

    print(metrics.classification_report(y_test, y_pred_rf))
    
    y_pred = clf.predict(x_train)
    print("Accuracy clf training set:", metrics.accuracy_score(y_train, y_pred))
    print(metrics.classification_report(y_train, y_pred))
    plot_learning_curves(clf, training.to_numpy(), target.to_numpy(), "Logistic Regression")

    return clf
