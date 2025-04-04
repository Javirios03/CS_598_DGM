import ml_collections

def create_config():
    config = ml_collections.ConfigDict()

    encoder = config.encoder = ml_collections.ConfigDict()
    encoder.input_dim = 2
    encoder.hidden_dim = 128
    encoder.output_dim = 2
    encoder.num_heads = 4
    encoder.depth = 4
    
    decoder = config.decoder = ml_collections.ConfigDict()
    decoder.hidden_dim = 128
    decoder.num_heads = 4
    decoder.conditioning_dim = 16
    decoder.depth = 4
    decoder.dropout = 0.0

    training = config.training = ml_collections.ConfigDict()
    training.lr = 1e-3
    training.epochs = 50
    training.weight_decay = 1e-2
    training.beta1 = 0.9
    training.beta2 = 0.999

    return config