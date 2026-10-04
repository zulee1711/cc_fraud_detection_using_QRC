SIMULATION = dict(
    n_customers=50,
    n_terminals=500,
    nb_days=365,
    start_date="2025-01-01",
    r=5,
    default_random_state=0,
)

customer_profiles, terminal_profiles, transactions = generate_dataset(**SIMULATION)
transactions = add_frauds(customer_profiles, terminal_profiles, transactions)
train, validation, test = split_dataset(transactions, train_ratio=0.70, validation_ratio=0.15)
