# src/schema_rules_unet.py

def build_unet_channels(base_channels=64, depth=4):
    """
    Returns encoder and decoder channel configurations.
    """
    encoder = [base_channels * (2 ** i) for i in range(depth)]
    bottleneck = base_channels * (2 ** depth)
    decoder = list(reversed(encoder))
    return encoder, bottleneck, decoder
