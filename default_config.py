class Config:
    init_scale = 0.05
    learning_rate = 0.1
    max_grad_norm = 5
    batch_size = 50
    sent_len = 100
    num_layers = 1
    keep_prob = 0.5
    vocab_size = None #Set when you load tok_map
    state_size = 256
    embed_size = 256
    num_models = 100
    num_epochs = 100
    debug = False
