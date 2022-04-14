import os 

# CLASSIFIER_PATH = "models/classification_8x8/{language}_{epochs}e.pth"
# CLASSIFIER_EVAL_PATH = "evaluation/classification_8x8/{language}_{epochs}e.pth"

# CLASSIFIER_IMBALANCE_POSITIVE_PATH = "models/classification_8x3/{language}_{epochs}e.pth"
# CLASSIFIER_IMBALANCE_POSITIVE_EVAL_PATH = "models/classification_8x3/{language}_{epochs}e.pth"
# CLASSIFIER_IMBALANCE_NEGATIVE_PATH = "models/classification_8x24/{language}_{epochs}e.pth"
# CLASSIFIER_IMBALANCE_NEGATIVE_EVAL_PATH = "models/classification_8x24/{language}_{epochs}e.pth"

# CLASSIFIER_VAR_SIZE_PATH = "models/variables_sizes_clf/{language}_{epochs}e_{max_filter_size}s_{n_filters}f_Acc.pth"

# classifier_multilingual_mode = lambda exclude_jap, multi_emb: ("_multiemb" if multi_emb else "") + ("_nojap" if exclude_jap else "")
# CLASSIFIER_MULTILINGUAL_PATH = "models/multi_classification/omni{mode}_{epochs}e.pth"
# CLASSIFIER_BILINGUAL_PATH = "models/multi_classification/bi_{language1}_{language2}_{epochs}e.pth"

# duplicate_label = lambda duplicate_id: '' if duplicate_id <= 0 else f'_v{duplicate_id}'
# REGRESSION_PATH = "models/regression/{language}_{epochs}e{duplicate_label}.pth"
# RETRIVAL_VECTORS_PATH = "models/regression/{language}_{epochs}e{duplicate_label}_vectors.txt"
# REGRESSION_EVAL_PATH = "evaluation/regression/{language}_{epochs}e{duplicate_label}_scores.pth"

RETRIEVAL_TMP = "/tmp/nn-morpho-analogy/ret"#regression
RETRIEVAL_CACHE = os.path.join(RETRIEVAL_TMP, "cache")
RETRIEVAL_VECTORS_PATH = os.path.join(RETRIEVAL_TMP, "{language}_{mode}_vectors.txt")

#REGRESSION_EUCLID_COSINE_EVAL_PATH = "evaluation/regression/euclid_cosine_r{search_range}_{epochs}e{duplicate_label}.csv"

#VOCAB_PATH = "vocab/{language}.txt"

def check_dir(path: str):
    if not os.path.exists(path):
        try:
            os.makedirs(path, exist_ok=True)
        except PermissionError:
            print(f"Permission error on missing '{path}', ignored. Please create it manually.")
check_dir('results')