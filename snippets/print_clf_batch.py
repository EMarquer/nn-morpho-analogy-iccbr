# %%
from statistics import mean, stdev
import pandas as pd
import torch
import pickle as pkl
# %%
root16 = "../3403636_clf-2016/"
root19 = "../3403636_clf-2019/"
records16 = pkl.load(open(root16+"results/clf.pkl","rb"))
records19 = pkl.load(open(root19+"results/clf.pkl","rb"))
records16
# %%
df16 = pd.DataFrame.from_records(records16, coerce_float=True).transpose()
df19 = pd.DataFrame.from_records(records19, coerce_float=True).transpose()
df16["best_model"] = root16 + df16["best_model"]
df19["best_model"] = root19 + df19["best_model"]
df = pd.concat([df16, df19], keys=["sig16+J", "sig19"])
df = df.rename_axis(['dataset', 'timestamp']).reset_index(level=0)
df = df.rename({"balacc": "bal. acc", "harmacc": "harm. acc"}, axis=1)
df
# %%
df["lang"] = df["lang"].apply(lambda x: x.capitalize())
df["bal. acc"] = df["bal. acc"].astype(float)
df["harm. acc"] = df["harm. acc"].astype(float)
df["TPR"] = df["TPR"].astype(float)
df["TNR"] = df["TNR"].astype(float)
df["F1"] = df["F1"].astype(float)
df["bal. acc %"] = df["bal. acc"].apply(lambda x: x*100).astype(float)
df["harm. acc %"] = df["harm. acc"].apply(lambda x: x*100).astype(float)
df["TPR %"] = df["TPR"].apply(lambda x: x*100).astype(float)
df["TNR %"] = df["TNR"].apply(lambda x: x*100).astype(float)
df["lang, dataset"] = df["dataset"] + " - " + df["lang"]
df = df.sort_index().drop_duplicates(subset=("seed","seed_id","lang","dataset"),keep='last')

# %%
x = df.groupby("lang, dataset")["bal. acc"].mean().sort_values()
# %%
import seaborn as sns
import matplotlib.pyplot as plt
min_acc=df["bal. acc %"].min()
max_acc=df["bal. acc %"].max()

plt.figure(figsize=(17,5))
ax = sns.barplot(data=df,x="lang, dataset",y="bal. acc %",ci="sd",order=reversed(x.index))
#ax.set_xticks([i for i, _ in enumerate(ax.get_xticklabels())], ax.get_xticklabels())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, horizontalalignment='right')
plt.axhline(y=max_acc, color='r', linestyle='--',label=f"max acc.: {max_acc:.2f}%")
plt.axhline(y=min_acc, color='r', linestyle='-',label=f"min acc.: {min_acc:.2f}%")
#ax.set_yticks(list(ax.get_yticks()) + [min_acc, max_acc])
ax.legend(loc="upper right")
ax.set_ylim((min_acc-1)//1,101)
ax.set_title("Classification balanced accuracy, in decreasing order of means. Error bars are standard deviation.")
None

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
