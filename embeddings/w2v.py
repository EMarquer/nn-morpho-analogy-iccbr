
import torch

def convert(path_in="embeddings/w2v/german-vectors.txt", path_out="embeddings/w2v/german.pth"):
    model = dict()
    with open("w2v/german-vectors.txt") as f:
        for l in f:
            word, vec = l.split(" ", 1)
            word = eval(word).decode(encoding="utf-8")
            vec = torch.tensor([float(x) for x in vec.strip().split(" ")])
            model[word]=vec
            
    torch.save(model, "w2v/german.pth")

class W2V:
    def __init__(self, language="german", path="embeddings/w2v/{}.pth") -> None:
        self.data = torch.load(path.format(language))

    def __getitem__(self, index):
        return self.data.get(index, self.data['UNK'])

if __name__ == "__main__":
    convert()
#m = W2V("w2v/german.pth")

