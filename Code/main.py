
import prediction
import controllo_llm as clm
#import query_prolog as qp se la import ottengo l'errore "list index out of range" su match = pattern.match(ret[-1]) in pyswip/core.py e non riesco a risolverlo
def main():
    
    
    # ESEMPIO DI ESECUZIONE
    while True:
        opzione = input("Scegli quale funzionalità vorresti usare!\n\
        1) Predizione di una transazione\n\
        2) Interroga la base di conoscenza\n")
        if opzione == 1 or opzione == 2:
            break
        
    
    if opzione == 1:
        # Predizione di un elemento effettuata su un caso eliminato dal dataset
        
        row = {'Type' : 'CASH_IN','Amount' : 7030.66,'OldBalanceOrig' : 14643510.38,'NewBalanceOrig' : 4650541.04,'OldBalanceDest' : 833165.14,'NewBalanceDest' : 826134.48}
        p = prediction.predict(row)
        
        #Per nuovi elementi, non estratti dal dataset
        '''
        type = input("Inserisci il tipo di transazione")
        amount = input("inserisci l'ammonto della transazione")
        OldBalanceOrig = input("inserisci il bilancio del mittente pre-transazione")
        NewBalanceOrig = input("inserisci il bilancio del mittente post-transazione")
        OldBalanceDest = input("inserisci il bilancio del destinatario pre-transazione")
        NewBalanceDest = input("inserisci il bilancio del destinatario post-transazione")
        row = {'Type' : type,'Amount' : amount,'OldBalanceOrig' : OldBalanceOrig,'NewBalanceOrig' : NewBalanceOrig,'OldBalanceDest' : OldBalanceDest,'NewBalanceDest' : NewBalanceDest}
        '''
        
        # Test del Large Language Model
        p = 'Ambiguo' #forzo ad ambiguo così da poter immediatamente testare la funzionalità
        if p == 'Ambiguo':
            print('''Sembra che la tua transazione non sia stata riconosciuta in modo adeguato.\n
                Per questo motivo, andremo a fare degli ulteriori controlli.\n
                Rimanga in attesa durante l'analisi della sua transazione''')
            msg = clm.controllo_llm(row)
            print(msg)
    else:
        opzione = -1
        while opzione not in [1, 2, 3, 4]:
            opzione = input(
                '''
                Cosa vorresti chiedere alla base di conoscenza?
                1) Se un utente X è socio di un utente Y
                2) Lista dei soci dell'utente X
                3) Lista delle transazioni fraudolente
                4) Lista dei Top N utenti più ricchi
                '''
            )
        if opzione == 1:
            print("Quali utenti vuoi ispezionare?")
            socio1 = input("Primo utente:")
            socio2 = input("Secondo utente:")
            query = f"soci({socio1.lower()}, {socio2})"
        elif opzione == 2:
            print("Quale utente vuoi ispezionare?")
            socio = input("Utente:")
            query = f"lista_soci({socio.lower()}, Y)"
        elif opzione == 3:
            query = "lista_cluster_fraudolenti(Lista)"
        elif opzione == 4:
            n_utenti = input("Inserisci il numero di utenti che vorresti nella top: ")
            query = f"utenti_piu_ricchi(Top, {n_utenti})"
        #qp.query_prolog(query) purtroppo la libreria non è funzionante sul mio pc quindi devo escludere la query, ottengo l'errore "list index out of range"
    input("Quando sei pronto a chiudere, premi un tasto qualsiasi.")
    print('Grazie per aver utilizzato il nostro sistema!')
            
    


if __name__ == '__main__':
    main() 