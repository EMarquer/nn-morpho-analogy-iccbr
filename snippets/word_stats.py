# %%
#import torch, torch.nn as nn
from siganalogies import dataset_factory
from siganalogies.config import SIG2016_LANGUAGES
from collections import Counter
from statistics import mean, stdev
import pandas as pd

def process_word(word):
    count = Counter(word)
    rate = mean(n/len(word) for n in count.values())
    adj = sum([word[i] == word[i + 1] for i in range(len(word) - 1)])

    return {"average repeated letters rate": rate, "number of adjacent repeated letters": adj, "length": len(word)}

print(process_word("naffaqa"))

# %%
for lang in SIG2016_LANGUAGES:
    dataset = dataset_factory(language=lang, mode="test")
    candidates = {i for pair in dataset.analogies for i in pair}

    records = []
    for a, _, b in [dataset.raw_data[i] for i in candidates]:
        records.append(process_word(a))
        records.append(process_word(b))

    df = pd.DataFrame.from_records(records)
    df.to_csv(f"stats/{lang}.csv")
    print(lang)
    print(df.describe())
# %%
dfs = [pd.read_csv(f"stats/{lang}.csv", index_col=0) for lang in SIG2016_LANGUAGES]
df = pd.concat(dfs, keys=SIG2016_LANGUAGES)
df["lang"] = [df.index[i][0] for i in range(len(df.index))]

df.groupby("lang").mean()
# %%
df.groupby("lang").max()
# %%
f = lambda x: f"{mean(x)}+-{stdev(x)} [{min(x)}; {max(x)}]"
print(df.groupby("lang").agg(["mean", "std", "max"]).to_latex())
