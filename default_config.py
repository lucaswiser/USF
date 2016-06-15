class TokRNNConfig:
    init_scale = 0.05
    learning_rate = 0.1
    max_grad_norm = 5
    batch_size = 512
    sent_len = 100
    num_layers = 1
    keep_prob = 0.5
    vocab_size = None #Set when you load tok_map
    state_size = 256
    embed_size = 256
    num_models = 100
    num_epochs = 100
    save_every = 10
    debug = False

class CharRNNConfig:
    init_scale = 0.05
    learning_rate = 0.1
    max_grad_norm = 5
    batch_size = 256
    sent_len = 500
    num_layers = 1
    keep_prob = 0.5
    vocab_size = None #Set when you load char_map
    state_size = 256
    embed_size = 64
    num_models = 100
    num_epochs = 100
    save_every = 5
    debug = False

class TokCNNConfig:
    init_scale = 0.05
    learning_rate = 0.1
    max_grad_norm = 5
    batch_size = 256
    sent_len = 100
    keep_prob = 0.5
    vocab_size = None #Set when you load char_map
    filter_sizes = [5]
    num_filters = [100]
    embed_size = 128
    num_models = 100
    num_epochs = 100
    save_every = 5
    debug = False