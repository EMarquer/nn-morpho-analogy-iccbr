# %%
import gzip
import torch, torch.nn as nn

# %%
def unpack():
    """Load the large GZIP file anf write 1 GZIP file per language"""
    lang = None
    analogies = dict()
    with gzip.open("lepage-sigmorphon.gz", 'r') as f:
        for line in f:
            if line.startswith(b'#'):
                lang = line[2:-2].decode("utf-8")
                print(f"new lang '{lang}'")
                analogies[lang] = []
            else:
                analogies[lang].append(line[:-1].decode("utf-8"))

    # %%
    for lang, analogies_lang in analogies.items():
        with gzip.open(f"lepage-sigmorphon-{lang}.gz", 'w') as f:
            f.write('\n'.join(analogies_lang).encode('utf-8'))

# %%
class LepageSigmorphonDataset(torch.utils.data.Dataset):
    def __init__(self, language="german"):
        super(LepageSigmorphonDataset).__init__()

        self.data = []
        with gzip.open(f"lepage-sigmorphon-{language}.gz", 'r') as f:
            for line in f:
                a,b,_,c,d = line.decode('utf-8').strip().split(':', maxsplit=5)
                self.data.append((a.strip(),b.strip(),c.strip(),d.strip()))

    def __len__(self): return len(self.data)
    def __getitem__(self, index): return self.data[index]

#data = LepageSigmorphonDataset("german")