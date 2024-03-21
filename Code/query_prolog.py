from pyswip import Prolog

def query_prolog(query):
    prolog = Prolog()
    prolog.consult('../Dataset/analisi_transazioni.pl')
    for res in prolog.query(query=query):
        print(res)