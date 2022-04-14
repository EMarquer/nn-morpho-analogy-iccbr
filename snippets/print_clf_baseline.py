# %%
import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR,'..'))
sys.path.insert(0, PARENT_DIR)
from statistics import mean, stdev
import pandas as pd
import torch
import pickle as pkl
from siganalogies import SIG2016_LANGUAGES
import matplotlib.pyplot as plt
import seaborn as sns
# %%
root16_am = "../3437442_baselines-50k-2016-alea-murena/results/baselines/"
root16_l = "../342_baselines-2016-lepage/results/baselines/"

def load(path, lang, model):
    df = pd.read_csv(path, index_col=0, converters={"fails":lambda x: x[1:-1].split(", "), "total":lambda x: x[1:-1].split(", ")})
    df["lang"] = lang
    df["model"] = model
    df["bal. acc"] = df["balacc"]
    df["bal. acc %"] = df["balacc"]*100
    df["harm. acc"] = df["harmacc"]
    df["harm. acc %"] = df["harmacc"]*100
    df["precision %"] = df["precision"]*100
    del df["balacc"]
    del df["harmacc"]
    if "fails" in df.columns:
        df["fails_pos"] = df["fails"].apply(lambda x: int(x[0]))
        df["fails_neg"] = df["fails"].apply(lambda x: int(x[1]))
        df["total_pos"] = df["total"].apply(lambda x: int(x[0]))
        df["total_neg"] = df["total"].apply(lambda x: int(x[1]))
        del df["fails"]
        del df["total"]
    return df
df_a = pd.concat(load(root16_am+f"{lang}/alea-0-50000.csv", lang, "alea") for lang in SIG2016_LANGUAGES)
df_m = pd.concat(load(root16_am+f"{lang}/murena-0-50000.csv", lang, "murena") for lang in SIG2016_LANGUAGES)
df_l = pd.concat(load(root16_l+f"{lang}/lepage-0-50000.csv", lang, "lepage") for lang in SIG2016_LANGUAGES)
df = pd.concat([df_a, df_m, df_l]).reset_index()
df
# %%
#df_ = df[["lang", "model", ]]
min_acc=df["bal. acc %"].min()
max_acc=df["bal. acc %"].max()
plt.figure(figsize=(17,5))
ax = sns.barplot(data=df, x="lang", y="bal. acc %", hue="model", ci="sd")
plt.axhline(y=max_acc, color='r', linestyle='--',label=f"max acc.: {max_acc:.2f}%")
plt.axhline(y=min_acc, color='r', linestyle='-',label=f"min acc.: {min_acc:.2f}%")
ax.legend(loc="upper right")
ax.set_ylim((min_acc-1)//1,101)


# %%

# df16 = pd.DataFrame.from_records(records16, coerce_float=True).transpose()
# df16["best_model"] = root16 + df16["best_model"]
# df16["model"] = "ours"
# df = pd.concat([df16], keys=["sig16+J", "sig19"])
# df = df.rename_axis(['dataset', 'timestamp']).reset_index(level=0)
# df = df.rename({"balacc": "bal. acc", "harmacc": "harm. acc"}, axis=1)
# df

# %%
plt.figure(figsize=(17,5))
ax = sns.barplot(data=df,x="lang, dataset",y="bal. acc %",ci="sd",order=sorted(x.index))
#ax.set_xticks([i for i, _ in enumerate(ax.get_xticklabels())], ax.get_xticklabels())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, horizontalalignment='right')
plt.axhline(y=max_acc, color='r', linestyle='--',label=f"max acc.: {max_acc:.2f}%")
plt.axhline(y=min_acc, color='r', linestyle='-',label=f"min acc.: {min_acc:.2f}%")
plt.axvline(x=10.5, color='k', linestyle='-')#,label=f"sep")
#ax.set_yticks(list(ax.get_yticks()) + [min_acc, max_acc])
ax.legend(loc="lower right")
ax.set_ylim((min_acc-1)//1,101)
ax.set_title("Classification balanced accuracy, in alphabetical order. Error bars are standard deviation.")

# %% To latex

y = df.groupby("lang")["acc"].aggregate(lambda x: f"${mean(x):5.02f} \pm {stdev(x):4.02f}$")
y_neg = df.groupby("lang")["acc_neg"].aggregate(lambda x: f"${mean(x):5.02f} \pm {stdev(x):4.02f}$")
y_pos = df.groupby("lang")["acc_pos"].aggregate(lambda x: f"${mean(x):5.02f} \pm {stdev(x):4.02f}$")
z = pd.DataFrame({"acc":y,"acc_pos":y_pos,"acc_neg":y_neg})
z.index = [k.capitalize() for k in z.index]
print(z.to_latex().replace("\$", "$").replace("\\textbackslash ","\\"))
# %%
df[df["lang"]=="german"]
# %%








# %% best results
import os

df["best_model"].apply(lambda x: os.path.dirname(x) + "/../failures.csv").to_list()
# %%
