# %%
from statistics import mean, stdev
from numpy import dtype
import pandas as pd
import torch
import pickle as pkl
import os, sys

# Change the current working directory
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from siganalogies import SIG2016_LANGUAGES
# %%
def load_lang_baseline(model="lepage",lang="arabic",size=50000):
    file = f"342_baselines-2016-lepage/results/baselines/{lang}/{model}-0-{size}.csv"
    df = pd.read_csv(file, index_col=0, dtype="float")
    df["lang"] = lang.capitalize()
    df["model"] = model
    del df["mrr"]
    return df
load_lang_baseline()
# %%
df16_alea = pd.concat([load_lang_baseline(lang=lang) for lang in SIG2016_LANGUAGES])
df16_alea["dataset"]="sig16+J"
df = df16_alea
df = df.rename({"balacc": "bal. acc", "harmacc": "harm. acc"}, axis=1)
df
# %%
df["precision (in %)"] = df["precision"].astype(float)*100
df["success@10"] = df["success@10"].astype(float)
df["lang, dataset"] = df["dataset"] + " - " + df["lang"]
#df = df.sort_index().drop_duplicates(subset=("seed","seed_id","lang","dataset"),keep='last')

# %%
import seaborn as sns
import matplotlib.pyplot as plt
value = "precision (in %)"
x = df.groupby("lang, dataset")[value].mean().sort_values()
min_value=df[value].min()
max_value=df[value].max()

plt.figure(figsize=(17,5))
ax = sns.barplot(data=df,x="lang, dataset",y=value,ci="sd",order=reversed(x.index))
#ax.set_xticks([i for i, _ in enumerate(ax.get_xticklabels())], ax.get_xticklabels())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, horizontalalignment='right')
plt.axhline(y=max_value, color='r', linestyle='--',label=f"max: {max_value:.2f}%")
plt.axhline(y=min_value, color='r', linestyle='-',label=f"min: {min_value:.2f}%")
#ax.set_yticks(list(ax.get_yticks()) + [min_acc, max_acc])
ax.legend(loc="upper right")
ax.set_ylim((min_value-1)//1,101)
ax.set_title(f"Baseline (lepage) classification {value}, in decreasing order of means. Error bars are standard deviation.")
None

# %%
plt.figure(figsize=(17,5))
ax = sns.barplot(data=df,x="lang, dataset",y=value,ci="sd",order=sorted(x.index))
#ax.set_xticks([i for i, _ in enumerate(ax.get_xticklabels())], ax.get_xticklabels())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, horizontalalignment='right')
plt.axhline(y=max_value, color='r', linestyle='--',label=f"max: {max_value:.2f}%")
plt.axhline(y=min_value, color='r', linestyle='-',label=f"min: {min_value:.2f}%")
plt.axvline(x=10.5, color='k', linestyle='-')#,label=f"sep")
#ax.set_yticks(list(ax.get_yticks()) + [min_acc, max_acc])
ax.legend(loc="upper right")
ax.set_ylim((min_value-1)//1,101)
ax.set_title(f"Baseline classification {value}, in alphabetical order. Error bars are standard deviation.")

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
