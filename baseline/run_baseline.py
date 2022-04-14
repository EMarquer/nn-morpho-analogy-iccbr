import os, sys
from statistics import mean

from numpy import iterable

# Change the current working directory
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(ROOT)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
logging.getLogger("").setLevel(logging.INFO)
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

def install_lepage_if_needed():
    """ From Lepage's README.txt
        - Extract the zip file (already done)
        - Open Terminal
        - Install
            >$ cd Nlg-1.0/ (we named this "lepage")
            >$ pip install .
    """
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    try:
        import baseline.lepage.nlg.Analogy.tests.nlg_benchmark as lepage
    except ModuleNotFoundError:
        logger.warning("Import error when loading Lepage's NLG, trying to install it.")
        import subprocess

        lepage_root = os.path.join(ROOT, "baseline", "lepage")
        # >$ cd Nlg-1.0/
        # >$ pip install .
        os.chdir(lepage_root)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "."])
    finally:
        os.chdir(ROOT)
install_lepage_if_needed()

import baseline.lepage.nlg.Analogy.tests.nlg_benchmark as lepage
import baseline.murena.analogy as murena
import baseline.alea.alea as alea
import pandas as pd
from multiprocessing import Pool

from utils.data import prepare_dataset
from utils import tpr_tnr_balacc_harmacc_f1
from siganalogies import SIG2016_LANGUAGES, SIG2019_HIGH, enrich, generate_negative, n_pos_n_neg

os.environ['PYTHONHASHSEED'] = str(42)
MODEL_RANDOM_SEEDS = [42, 8564851, 706303, 248, 8994204, 7332146, 800, 3347863, 1402754, 7938707]
RHO = 100
"""
source ~/miniconda3/bin/activate nn-morpho-analogy ; cd ~/orpailleur/emarquer/nn-morpho-analogy ; python run_baseline.py arabic murena
"""

def lepage_wrapper(a,b,c,d):
    result = lepage.is_analogy(f"{a} : {b} :: {c} : {d}")
    if result:
        return True, 0
    else:
        return False, 99999999

### multiprocessing wrappers ###
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired, concurrent
import multiprocessing
#multiprocessing.set_start_method("spawn")

def process_one_perm(a,b=None,c=None,d=None, model_name="murena", n_max=10):
    if b is None: a,b,c,d = a
    if model_name == "murena":
        if STORE_PREDS:
            # format:
            # rank (0 to n_max-1) or 99999999, predictions
            return murena.classification_with_results(a,b,c,d, n_max)[1:]
        else:
            # format:
            # rank (0 to n_max-1)
            return murena.classification_(a,b,c,d, n_max)[1]
    
    elif model_name == "alea":
        if STORE_PREDS:
            # format:
            # rank (0 to n_max-1) or 99999999, predictions
            return alea.classification_with_results(a,b,c,d, n_max, rho=RHO)[1:]
        else:
            # format:
            # rank (0 to n_max-1)
            return alea.classification_(a,b,c,d, n_max, rho=RHO)[1]
    elif model_name == "lepage":
        # format:
        # 0 or 99999999
        return lepage_wrapper(a,b,c,d)[1]

def failed_process():
    tps, tns, fns, fps, rs = 0, 0, 8*4, 8*4, [99999999] * 8 * 4
    return tps, tns, fns, fps, rs

from siganalogies.abstract_analogy_dataset import AbstractAnalogyDataset
from siganalogies.encoders import NO_ENCODER
from functools import partial
from tqdm import tqdm # for progress bar

def run_multiprocessing(model_name, n_max, test_data, num_processes=multiprocessing.cpu_count(), timeout=5, error_as_failure=True, progress_bar=False):
    pos, neg = zip(*[n_pos_n_neg(a, b, c, d, filter_invalid=False, tensors=False, n=None) for a, b, c, d in test_data])
    pos = sum(pos, start=[])
    neg = sum(neg, start=[])

    tps, tns, fns, fps, rs = 0, 0, 0, 0, []
    
    fails_global = []
    total_global = []
    for (positive, iterable) in [(True, pos), (False, neg)]:
        with ProcessPool(max_workers=num_processes) as pool:
            future = pool.map(partial(process_one_perm, model_name=model_name, n_max=n_max), iterable, timeout=timeout)
            it = future.result()
            pool.close()
        
            fails = 0
            total = len(iterable)
            if progress_bar: pbar = tqdm(total=total, desc="positive" if positive else "negative")
            while True:
                r = None
                try:
                    r = next(it)
                except StopIteration:
                    break
                except TimeoutError as error:
                    #logger.error("Function took longer than %d seconds" % error.args[1])
                    if error_as_failure: r = 99999999
                    fails += 1
                except ProcessExpired as error:
                    #logger.error("%s. Exit code: %d" % (error, error.exitcode))
                    if error_as_failure: r = 99999999
                    fails += 1
                except Exception as error:
                    logger.error("Function raised %s" % error)
                    logger.error(error.traceback)  # Python's traceback of remote process
                    fails += 1

                if r is not None:
                    if positive:
                        rs.append(r)
                        tps += 1 if r < n_max else 0
                        fns += 1 if r >= n_max else 0

                    else:
                        tns += 1 if r > n_max - 1 else 0
                        fps += 1 if r <= n_max - 1 else 0

                if progress_bar: pbar.update()
                if progress_bar: pbar.set_postfix({"failures": fails})
            if progress_bar: pbar.close()
            #pool.close()
        fails_global.append(fails)
        total_global.append(total)
    return tps, tns, fns, fps, rs, fails_global, total_global

def precision_sak_ap_rr(r_from_0, k=10):
    """Computes: precision, success@k, rank, reciprocal rank, closest word to prediction, closest word to target"""
    precision = 1. if r_from_0==0 else 0.
    sak = 1. if r_from_0<k else 0.
    r = r_from_0 + 1
    rr = 1/r

    return precision, sak, r, rr

STORE_PREDS = False
def run_model(model_name, test_data, n_max=10, progress_bar=False):
    """All precision measures are tested at k=10 for positive samples and k=1 for negative samples"""

    p = Pool(processes=4)
    logger.info("Starting processing of the test data:")
    tp, tn, fn, fp, rs, fails, total = run_multiprocessing(model_name=model_name, n_max=n_max, test_data=test_data, num_processes=args.processes, timeout=args.timeout, progress_bar=progress_bar)
    logger.info("Done!")

    # classification results
    tpr, tnr, balacc, harmacc, f1 = tpr_tnr_balacc_harmacc_f1(tp, tn, fp, fn)

    # retrieval results
    precision, sak, r, rr = zip(*(precision_sak_ap_rr(rank) for rank in rs))
    m_p, m_sak, m_r, m_rr = mean(precision), mean(sak), mean(r), mean(rr)
    return {
        "TPR": tpr,
        "TNR": tnr,
        "balacc": balacc,
        "harmacc": harmacc,
        "F1": f1,
        "fails": fails,
        "total": total,

        "precision": m_p,
        f"success@{n_max}": m_sak,
        "mrr": m_rr, # /!\ limited to top n_max (usually the top 10)
    }, (tpr, tnr, balacc, harmacc, f1), (m_p, m_sak, m_r, m_rr)


def main(args):
    logger.info(f"Processing baseline {args.model} on {args.language}...")
    train_data, val_data, test_data, dataset = prepare_dataset(
        args.dataset, args.language, args.nb_analogies_train, args.nb_analogies_val, args.nb_analogies_test,
        args.force_rebuild, split_seed=MODEL_RANDOM_SEEDS[args.version])

    def dataset_to_no_char_but_no_rebuild(dataset: AbstractAnalogyDataset):
        dataset.word_encoder = NO_ENCODER
    dataset_to_no_char_but_no_rebuild(train_data.dataset)
    dataset_to_no_char_but_no_rebuild(val_data.dataset)
    dataset_to_no_char_but_no_rebuild(test_data.dataset)
    dataset_to_no_char_but_no_rebuild(dataset)

    x, y = test_data.dataset.analogies[test_data.indices[2]]
    print(test_data.dataset.raw_data[x])
    print(test_data.dataset.raw_data[y])

    record, (tpr, tnr, balacc, harmacc, f1), (m_p, m_sak, m_r, m_rr) = run_model(args.model,test_data, n_max=args.n_max, progress_bar=args.progress_bar)

    logger.info(f"Baseline {args.model} on {args.language}:\n{record}.")
    os.makedirs(f"results/baselines/{args.language}", exist_ok=True)
    pd.DataFrame.from_records([record]).to_csv(f"results/baselines/{args.language}/{args.model}-{args.version}-{args.nb_analogies_test}.csv")
    logger.info(f"Processing baseline {args.model} on {args.language} done.")

if __name__ == '__main__':
    from argparse import ArgumentParser

    # argument parsing
    parser = ArgumentParser()
    model_parser = parser.add_argument_group("Model arguments")
    model_parser.add_argument('--model', '-m', type=str, default="all", help='The baseline model to use.', choices=["all", "lepage", "murena", "alea"])
    model_parser.add_argument('--n-max', '-N', type=int, default=1)

    dataset_parser = parser.add_argument_group("Dataset arguments")
    dataset_parser.add_argument('--dataset', '-d', type=str, default="2016", help='The language to train the model on.', choices=["2016", "2019"])
    dataset_parser.add_argument('--language', '-l', type=str, default="arabic", help='The language to train the model on.', choices=SIG2019_HIGH+SIG2016_LANGUAGES)
    dataset_parser.add_argument('--force-rebuild', help='Force the re-building of the dataset file.',  action='store_true')
    dataset_parser.add_argument('--recompute', help='Force the computation of the result even if already computed (combination version, nb-analogies-test, language, model) of the dataset file.',  action='store_true')
    dataset_parser.add_argument('--nb-analogies-train', '-n', type=int, default=50000, help='The maximum number of analogies (before augmentation) we train the model on.')
    dataset_parser.add_argument('--nb-analogies-val', '-v', type=int, default=500, help='The maximum number of analogies (before augmentation) we validate the model on.')
    dataset_parser.add_argument('--nb-analogies-test', '-t', type=int, default=50000, help='The maximum number of analogies (before augmentation) we test the model on.')
    dataset_parser.add_argument('--version', '-V', type=int, default=0, help='The experiment version, also corresponding to the random seed to use.')
    dataset_parser.add_argument('--processes', '-p', type=int, default=-1, help='The maximum number of processes to use.')
    dataset_parser.add_argument('--progress-bar', help='Display a progress bar.', action='store_true')
    dataset_parser.add_argument('--timeout', '-o', type=int, default=10, help='The maximum time per processes.')

    args = parser.parse_args()

    if args.processes < 0: args.processes = multiprocessing.cpu_count()

    if args.model == "all":
        for model in ["lepage", "murena", "alea"]:
            args.model = model
            if not os.path.exists(f"results/baselines/{args.language}/{args.model}-{args.version}-{args.nb_analogies_test}.csv") or args.recompute:
                main(args)
    else:
        if not os.path.exists(f"results/baselines/{args.language}/{args.model}-{args.version}-{args.nb_analogies_test}.csv") or args.recompute:
            main(args)
