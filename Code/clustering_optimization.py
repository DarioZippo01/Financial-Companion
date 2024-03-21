import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd



def bic_method(X):
    valori_bic = []
    for i in range(1, 13):
        em = GaussianMixture(n_components=i, random_state=42, reg_covar=1e-6, covariance_type='diag')
        em.fit(X)
        valori_bic.append(em.bic(X))
    #Cerchiamo quale sia il numero di cluster che ci porta ad avere un numero BIC minore
    k = np.argmin(valori_bic) + 1
    print(f'Il numero di cluster ottimale per il metodo di Expectation Maximization è {k}')
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 13), valori_bic, marker='o')
    plt.title('BIC per diversi numeri di cluster')
    plt.xlabel('Numero di Cluster')
    plt.ylabel('BIC')
    plt.xticks(range(1, 13))
    plt.show()
    

def em_clustering(X, orig):
    em = GaussianMixture(n_components=10, random_state=42, reg_covar=1e-6, covariance_type='diag')
    em.fit(X)
    cluster = pd.Series(em.predict(X))
    orig['Cluster'] = cluster
    return orig