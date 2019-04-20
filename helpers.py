from copy import deepcopy


def clone_layer(layer, num_clones):
    return [deepcopy(layer) for i in num_clones]


def generate_word_dict(pathToEnglishData, pathToFrenchData):
    word_dict = {}
    with open('data/train/giga-fren.release2.fixed.en', 'r') as file:
        line = file.readline()
        while line:
            # Process data
