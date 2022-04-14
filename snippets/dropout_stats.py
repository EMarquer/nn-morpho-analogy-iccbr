
from siganalogies.config import SIG2016_LANGUAGES
import pandas as pd
from scipy.stats import ttest_ind, ttest_1samp
import seaborn as sns
import matplotlib.pyplot as plt



DROPOUTS = [0.01, 0.05, 0.1, 0.3]
dfs = {
    is_valid:
        {dropout: pd.read_csv(f"dropout/evaluation_{dropout}_10times_{'' if is_valid else 'in'}valid_MSTD.csv", index_col=0) for dropout in DROPOUTS}
    for is_valid in [True, False]}

dfs[True][0] = pd.read_csv(f"dropout/valid.csv", index_col=0, sep=';')*100
dfs[False][0] = pd.read_csv(f"dropout/invalid.csv", index_col=0, sep=';')*100


def ttest(dfs, dropout_1, dropout_2, language, is_valid=True):
    if dropout_1 == 0:
        index = [f"{'' if is_valid else 'in'}valid_{i}" for i in range(10)]
        data_1 = dfs[is_valid][0]["classification_8"].loc[language]
        data_2 = dfs[is_valid][dropout_2][index].loc[language]
        
        return ttest_1samp(data_2, data_1).pvalue

    elif dropout_2 == 0:
        return ttest(dfs, dropout_2, dropout_1, language, is_valid)
    else:
        index = [f"{'' if is_valid else 'in'}valid_{i}" for i in range(10)]
        data_1 = dfs[is_valid][dropout_1][index].loc[language]
        data_2 = dfs[is_valid][dropout_2][index].loc[language]

        return ttest_ind(data_1, data_2).pvalue

def mean_std(dfs, dropout, language, is_valid=True):
    index = [f"{'' if is_valid else 'in'}valid_{i}" for i in range(10)]
    data = dfs[is_valid][dropout][index].loc[language]

    return data.mean(), data.std()

def format(mean, std, pvalue):
    if pvalue > 0.05: # non significative
        pvalue_string = ""
    elif pvalue > 0.01: # significative
        pvalue_string = "*"
    elif pvalue <= 0.01: # very significative
        pvalue_string = "**"

    return f"{mean} ± {std}{pvalue_string}"

# between 0.01 (instead of 0, placeholder)
records_valid = []
records_invalid = []
for language in SIG2016_LANGUAGES:
    record_valid = {"language": language, "no dropout": dfs[True][0]["classification_8"].loc[language].item()}
    record_invalid = {"language": language, "no dropout": dfs[False][0]["classification_8"].loc[language].item()}
    for i, dropout in enumerate(DROPOUTS):
        reference = ([0] + DROPOUTS)[i]
        pvalue = ttest(dfs, reference, dropout, language, is_valid=True)
        #records_valid.append({"language": language, "dropout": dropout, "mean ± std": format(*mean_std(dfs, dropout, language, is_valid=True), pvalue)})
        record_valid[f"{dropout} (mean ± std)"] = format(*mean_std(dfs, dropout, language, is_valid=True), pvalue)

        pvalue = ttest(dfs, reference, dropout, language, is_valid=False)
        #records_invalid.append({"language": language, "dropout": dropout, "mean ± std": format(*mean_std(dfs, dropout, language, is_valid=False), pvalue)})
        record_invalid[f"{dropout} (mean ± std)"] = format(*mean_std(dfs, dropout, language, is_valid=False), pvalue)
    records_valid.append(record_valid)
    records_invalid.append(record_invalid)

df_valid = pd.DataFrame.from_records(records_valid, index="language")
df_invalid = pd.DataFrame.from_records(records_invalid, index="language")

#
#df_valid

#df_invalid

print("Valid analogies; ttest_ind wrt the previous column (or ttest_1samp for 0.01 wrt. no dropout):\n\tnothing: pvalue> 0.05\n\t*:      0.05 > pvalue > 0.01\n\t**:      0.01 > pvalue)")
print(df_valid)

print()
print("Inalid analogies\n\tnothing: pvalue> 0.05\n\t*:      0.05 > pvalue > 0.01\n\t**:      0.01 > pvalue)")
print(df_invalid)



BASELINE_VALID = {
    "arabic": 32.94,
    "finnish": 25.6,
    "georgian": 89.78,
    "german": 85.46,
    "hungarian": 35.79,
    "japanese": 18.62,
    "maltese": 74.49,
    "navajo": 17.97,
    "russian": 42.,
    "spanish": 85.23,
    "turkish": 42.53
}
BASELINE_INVALID = {
    "arabic": 97.79,
    "finnish": 98.78,
    "georgian": 95.21,
    "german": 97.19,
    "hungarian": 98.40,
    "maltese": 69.29,
    "navajo": 94.93,
    "russian": 93.88,
    "spanish": 86.62,
    "turkish": 91.40,
    "japanese": 98.13,
}
records = []
for language in SIG2016_LANGUAGES:
    for is_valid in {True, False}:
        record = {
            "Language": language.capitalize(),
            "Model": "ANNc no dropout",
            "Analogy": f"{'V' if is_valid else 'Inv'}alid",
            "Average accuracy (in %) on run": dfs[is_valid][0]["classification_8"].loc[language].item()}
        records.append(record)
        for i, dropout in enumerate(DROPOUTS):
            index = [f"{'' if is_valid else 'in'}valid_{i}" for i in range(10)]
            data = dfs[is_valid][dropout][index].loc[language]
            record = {
                "Language": language.capitalize(),
                "Model": f"ANNc dropout {dropout}" if dropout>0 else "ANNc no dropout",
                "Analogy": f"{'V' if is_valid else 'Inv'}alid"
            }
            for m in data:
                records.append({**record, "Average accuracy (in %) on run": m})
    records.append({
        "Language": language.capitalize(),
        "Model": "Best baseline",
        "Analogy": "Valid",
        "Average accuracy (in %) on run": BASELINE_VALID[language]})
    records.append({
        "Language": language.capitalize(),
        "Model": "Best baseline",
        "Analogy": "Invalid",
        "Average accuracy (in %) on run": BASELINE_INVALID[language]})
df = pd.DataFrame.from_records(records)
        

cmap = plt.get_cmap("Greens_r")
colors = {
    f"ANNc dropout {dropout}" if dropout>0 else "ANNc no dropout": cmap((i+3)/11)
    for i, dropout in enumerate([0,0.01,0.05,0.1,.3])
}
colors["Best baseline"] = "b"
#plt.
order = ["Best baseline"] + [f"ANNc dropout {dropout}" if dropout>0 else "ANNc no dropout" for dropout in [0,0.01,0.05,0.1,.3]]
sns.catplot(data=df, x="Language", y="Average accuracy (in %) on run", kind="bar", ci="sd", hue="Model", aspect=4.5, palette=colors, hue_order=order,height=2.5,row="Analogy", row_order=["Valid", "Invalid"])





