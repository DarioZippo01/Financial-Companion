import pandas as pd
import numpy as np
from sklearn import preprocessing
import classification_function as cf

dizionario_types = {'PAYMENT' : 0, 'CASH_OUT' : 1, 'CASH_IN' : 2, 'DEBIT' : 3, 'TRANSFER' : 4}
dizionario_frodi = {0 : 'Genuino', 1 : 'Ambiguo', 2 : 'Frode'}
def predict(row_utente):
    transazioni = pd.read_csv('../Dataset/dataset_preprocessato.csv')
    transazioni = transazioni.drop(columns=["NameDest", "NameOrig", 'Type'])

    target = transazioni["IsFraud"]
    training = transazioni.drop(columns=["IsFraud"])  
    scaler = preprocessing.StandardScaler()
    scaled_df = scaler.fit_transform(training)
    training = pd.DataFrame(scaled_df)
    #Assegna indici ai valori inseriti dall'utente
    row_utente[1] = dizionario_types.get(row_utente[1], None)
    print(row_utente)
    rf = cf.random_forest_classification(training, target)
    classification_predict = rf.predict([row_utente])
    print(classification_predict)
    final_val = dizionario_frodi.get(classification_predict[0], None)
    print(f"La transazione è probabilmente {final_val} con probabilità: ", rf.predict_proba([row_utente]))