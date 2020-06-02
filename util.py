def connect(input_layer, layers):
    """ Connect the given sequence of layers and returns output layer

    Parameters
    ----------
    input_layer : tf.keras layer
        Input layer
    layers : list of tf.keras layers
        Layers to be connected sequentially

    Returns
    -------
    output_layer : tf.kears layer
        Output Layer
    """
    layer = input_layer
    for l in layers:
        layer = l(layer)
    return layer
