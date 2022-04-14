
from fasttext import load_model
import torch


class FastText:
    def __init__(self, language="german", path="embeddings/fasttext/{}.bin") -> None:
        self.model = load_model(path.format(language))

    def __getitem__(self, index):
        return torch.tensor(self.model.get_word_vector(index)).view(-1)


if False:
    
    print(model.words)
    print(model.labels)

    
    model.get_word_vector()

    
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import data
    siganalogies.PATH = "../sigmorphon2016/data/"
    dataset = siganalogies.Task1Dataset('german', "test", "none")

    
    for w in dataset.all_words:
        print(w)
        if (w in model.words):
            print(model.get_word_vector(w))
            break

    
    # coverage: 0.4917564520170383 for test
    from statistics import mean
    mean(1 if w in model.words else 0 for w in dataset.all_words)
    
