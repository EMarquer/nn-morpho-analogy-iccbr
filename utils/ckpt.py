import pickle as pkl
import pandas as pd
from os.path import join
from functools import partial

PREFIXES = {"clf": {
    "2016": "1_gathered-results"
}}
EXCLUDED_LANGS = ["basque", "uzbek"]

def get_clf_in_prefix(dataset, prefix=None):
    if not prefix: prefix = PREFIXES["clf"][dataset]
    file = join(prefix, "results/clf.pkl")
    with open(file, 'rb') as f:
        data = pkl.load(f)

    df = pd.DataFrame.from_records(data, coerce_float=True).transpose()
    df["best_model"] = df["best_model"].apply(partial(join, prefix))
    df = df.rename_axis(['timestamp']).reset_index(level=0)
    df["dataset, lang, version"] = dataset + " - " + df["lang"] + " - " + df["seed_id"].apply(str)
    df = df.sort_index().drop_duplicates(subset=("seed","dataset, lang, version"),keep='last')
    #print(df["dataset, lang, version"].unique().size / len(df["dataset, lang, version"]))
    df.index = df["dataset, lang, version"]
    return df["best_model"]