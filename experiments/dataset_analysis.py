import pandas as pd

import qrc

from qrc.datasets import split_dataset
from qrc.datasets.create_dataset import add_frauds, generate_dataset
#%%
#customers = [10,50,100,500,1000]
customers = [500,1000,2500,5000]
terminals = [5000,10000,25000,50000]

#Table comparing datasets generated with diff number of transactions and terminals

SIMULATION = dict(
    nb_days=365,
    start_date="2025-01-01",
    r=5,
    default_random_state=0,
)
#By default does analysis on train dataset 
def fraud_rate(customers=customers, terminals=terminals, simulation=SIMULATION):
    rows = []
    for n_customers in customers:
        for n_terminals in terminals:
            customer_profiles, terminal_profiles, transactions = generate_dataset(n_customers=n_customers,n_terminals=n_terminals,**SIMULATION)
            transactions = add_frauds(customer_profiles, terminal_profiles, transactions)
            train, validation, test = split_dataset(transactions, train_ratio=0.70, validation_ratio=0.15)
            
            fraud_rate = train["TX_FRAUD"].mean()
            rows.append({"n_customers":n_customers,
                         "n_terminals":n_terminals,
                         "fraud_rate": f"{fraud_rate * 100:.2f}%"})
    table = pd.DataFrame(rows).pivot(index="n_customers",columns="n_terminals",values = "fraud_rate")
    return table



