from copy import deepcopy


def clone_layer(layer, num_clones):
    return [deepcopy(layer) for i in num_clones]
