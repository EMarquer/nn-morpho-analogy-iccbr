# %%
import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR,'..'))
sys.path.insert(0, PARENT_DIR)
from siganalogies import dataset_factory_2019, dataset_factory_2016
from siganalogies import SIG2019_HIGH, SIG2019_LOW, SIG2019_LOW_MODES, SIG2019_HIGH_MODES
from siganalogies import SIG2016_LANGUAGES, SIG2016_MODES
import pandas as pd

# %%
records = {
        "#analogies": dict(),
        "#features with analogies (% of all features)": dict(),
        "#words with analogies (% of vocabulary)": dict(),
    }

l = [l_ for l_ in SIG2016_LANGUAGES if l_ != "japanese"]
for lang in l:
    dataset = dataset_factory_2016(lang, mode="test")
    lang=lang.capitalize()
    records["#analogies"][lang] = f"${len(dataset.analogies)}$"
    records["#features with analogies (% of all features)"][lang] = f"${len(dataset.features_with_analogies)}$ (${len(dataset.features_with_analogies)/len(dataset.features):.2%}$)"
    records["#words with analogies (% of vocabulary)"][lang] = f"${len(dataset.words_with_analogies)}$ (${len(dataset.words_with_analogies)/len(dataset.word_voc):.2%}$)"
df = pd.DataFrame.from_records(records)
df.index.name = "Language"
df
# %%
print(df.to_latex().replace("\$", "$"))
# %%
print(', '.join(sorted([c.capitalize() for c in set(SIG2019_HIGH) & set(SIG2019_LOW)])))
# %%
print(', '.join(sorted([c.capitalize() for c in set(SIG2019_HIGH) & set(SIG2016_LANGUAGES)])))
print(', '.join(sorted([c.capitalize() for c in set(SIG2016_LANGUAGES) & set(SIG2019_LOW)])))
# %%
dfs = dict()
for set_ in SIG2019_LOW_MODES:
    print()
    records = {
            "#analogies": dict(),
            "#features with analogies (% of all features)": dict(),
            "#words with analogies (% of vocabulary)": dict(),
        }
    for lang in SIG2019_LOW:
        dataset = dataset_factory_2019(lang, set_)
        lang='-'.join([l.capitalize() for l in lang.split('-')])
        records["#analogies"][lang] = f"${len(dataset.analogies)}$"
        records["#features with analogies (% of all features)"][lang] = f"${len(dataset.features_with_analogies)}$ (${len(dataset.features_with_analogies)/len(dataset.features):.2%}$)"
        records["#words with analogies (% of vocabulary)"][lang] = f"${len(dataset.words_with_analogies)}$ (${len(dataset.words_with_analogies)/len(dataset.word_voc):.2%}$)"
    df = pd.DataFrame.from_records(records)
    df.index.name = "Language"
    dfs[set] = df
# %%
dfs["train-low"]
print(dfs["train-low"].to_latex().replace("\$", "$"))
# %%
dfs["test"]
print(dfs["test"].to_latex().replace("\$", "$"))
# %%
dfs["dev"]
print(dfs["dev"].to_latex().replace("\$", "$"))
# %%
datasets = {lang.capitalize(): dataset_factory_2019(lang) for lang in SIG2019_HIGH}
# %%
records_features = dict()
records_analogies = dict()
for i, lang1 in enumerate(SIG2019_HIGH[:-1]):
    lang1=lang1.capitalize()
    records_features[lang1] = dict()
    records_analogies[lang1] = dict()
    for lang2 in SIG2019_HIGH[i+1:]:
        print(lang1, lang2)
        lang2=lang2.capitalize()
        shared_features = set(datasets[lang1].features) & set(datasets[lang2].features)
        records_features[lang1][lang2] = len(shared_features)
        if len(shared_features) > 0:
            l1 = [(i, (w1, f, w2)) for i, (w1, w2, f) in enumerate(datasets[lang1].raw_data) if f in shared_features]
            l2 = [(i, (w1, f, w2)) for i, (w1, w2, f) in enumerate(datasets[lang2].raw_data) if f in shared_features]
            print(len(l1), len(l2))
            analogies = [(i, j) 
                for i, (w11, f1, w12) in l1
                for j, (w21, f2, w22) in l2
                if f1 == f2]
            records_analogies[lang1][lang2] = len(analogies)
# %%
records_analogies = {k:v for k, v in records_analogies.items() if v}
df_features = pd.DataFrame.from_records(records_features)
df_analogies = pd.DataFrame.from_records(records_analogies)
# %%
import seaborn as sns
sns.heatmap(data=df_features)
# %%
sns.heatmap(data=df_analogies)

# %%
from copy import copy, deepcopy
import matplotlib.pyplot as plt
records_analogies_ = deepcopy(records_analogies)
for l1, dict_ in records_analogies.items():
    for l2, v in dict_.items():
        dict__ = records_analogies_.get(l2, dict())
        dict__[l1] = v
        records_analogies_[l2] = dict__
df_analogies_ = pd.DataFrame.from_records(records_analogies_)
df_analogies_ = df_analogies_.sort_index().reindex(sorted(df_analogies_.columns), axis=1)
plt.figure(figsize=(15,13))
sns.heatmap(data=df_analogies_, cmap="coolwarm")
# %%
from matplotlib.colors import LogNorm
plt.figure(figsize=(15,13))
sns.heatmap(data=df_analogies_, cmap="coolwarm", square=True, norm=LogNorm())

# %%
i = 0
flat = {"count": dict()}
for l1, dict_ in records_analogies.items():
    for l2, v in dict_.items():
        #print(f"{l1} & {l2}")
        i += 1
        flat["count"][(l1,l2)] = v
print(i)
df_flat = pd.DataFrame.from_records(flat)
# %%

g = sns.boxplot(y="count", data=df_flat)
g.set_yscale("log")
g.set_ylim(bottom=0)
# %%
g = sns.violinplot(y="count", data=df_flat)

# %%
(df_flat[df_flat["count"] > 0]).max()
# %%
