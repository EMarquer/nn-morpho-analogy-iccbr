import pickle
from functools import partial
import logging
from packaging import version
logger = logging.getLogger("")#__name__)
logger.setLevel(logging.INFO)


# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add the handlers to the loggers
logger.addHandler(ch)

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
# configure logging at the root level of lightning
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

import config
from analogy_clf import AnalogyClassification
from cnn_embeddings import CNNEmbedding
from siganalogies import SIG2016_LANGUAGES, SIG2019_HIGH, enrich as enrich_cp, generate_negative as generate_negative_cp, n_pos_n_neg as n_pos_n_neg_cp, enrich_no_cp, generate_negative_no_cp, n_pos_n_neg_no_cp
from utils import prepare_data, tpr_tnr_balacc_harmacc_f1, CSVCustomLogger, mask_valid
from pytorch_lightning.plugins import DDPSpawnPlugin, DDPPlugin, DDP2Plugin

import os
os.environ['PYTHONHASHSEED'] = str(42)
seed_everything(42, workers=True)

MODEL_RANDOM_SEEDS = [42, 8564851, 706303, 248, 8994204, 7332146, 800, 3347863, 1402754, 7938707]
USE_STRATEGY = version.parse(pl.__version__) >= version.parse("1.5.0")
USE_SPAWN = False


# --- for the custom logger ---
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only

# --- for the confusion matrices
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

class CPnoCPLogger(LightningLoggerBase):
    @property
    def name(self):
        return "MyLogger"

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step=0):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        if (
            not (hasattr(self, "tp") and hasattr(self, "tn") and hasattr(self, "fp") and hasattr(self, "fn")) or
            (self.tp is None or self.tn is None or self.fp is None or self.fn is None)
            ):
            self.tp = torch.zeros_like(metrics['true_positive'])
            self.tn = torch.zeros_like(metrics['true_positive'])
            self.fp = torch.zeros_like(metrics['true_positive'])
            self.fn = torch.zeros_like(metrics['true_positive'])
        if {'true_positive', 'true_negative', 'false_positive', 'false_negative'}.issubset(metrics.keys()):
            self.tp += metrics['true_positive']
            self.tn += metrics['true_negative']
            self.fp += metrics['false_positive']
            self.fn += metrics['false_negative']

    def get_metrics(self):
        # don't show the version number
        tpr, tnr, balacc, harmacc, f1 = tpr_tnr_balacc_harmacc_f1(self.tp, self.tn, self.fp, self.fn)
        return {"tpr": tpr, "tnr": tnr, "balacc": balacc, "harmacc": harmacc, "f1": f1}

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        # If you implement this, remember to call `super().save()`
        # at the start of the method (important for aggregation of metrics)
        super().save()

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        self.tpr, self.tnr, self.balacc, self.harmacc, self.f1 = tpr_tnr_balacc_harmacc_f1(self.tp, self.tn, self.fp, self.fn)
        pass

class ClfLightning(pl.LightningModule):
    def __init__(self, char_emb_size, encoder, filters = 128, drop_fake_negative=False, cp = ""):
        super().__init__()
        self.save_hyperparameters()
        self.emb = CNNEmbedding(voc_size=len(encoder.id_to_char), char_emb_size=char_emb_size)
        self.clf = AnalogyClassification(emb_size=self.emb.get_emb_size(), filters = filters)
        self.encoder = encoder
        
        self.criterion = nn.BCELoss()

        self.drop_fake_negative = drop_fake_negative

        self.cp = cp

    def configure_optimizers(self):
        # @lightning method
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        if self.cp == "undef":
            n_pos_n_neg = n_pos_n_neg_no_cp
        elif self.cp == "bad":
            n_pos_n_neg = partial(n_pos_n_neg_no_cp, cp_undefined=False)
        else:
            n_pos_n_neg = n_pos_n_neg_cp
    
        # @lightning method
        a,b,c,d = batch

        # compute the embeddings
        a = self.emb(a)
        b = self.emb(b)
        c = self.emb(c)
        d = self.emb(d)

        loss = torch.tensor(0, device=a.device, dtype=float)
        pos, neg = [], []

        pos_permutations, neg_permutations = n_pos_n_neg(a, b, c, d, filter_invalid=True)

        # positive example, target is 1
        for a_, b_, c_, d_ in pos_permutations:
            is_analogy = self.clf(a_, b_, c_, d_)

            expected = torch.ones(is_analogy.size(), device=is_analogy.device)
            loss += self.criterion(is_analogy, expected)
            pos.append(is_analogy >= 0.5)

        # negative example, target is 0
        for a_, b_, c_, d_ in neg_permutations:
            is_analogy = self.clf(a_, b_, c_, d_)

            expected = torch.zeros(is_analogy.size(), device=is_analogy.device)
            loss += self.criterion(is_analogy, expected)
            neg.append(is_analogy < 0.5)

        pos = torch.cat(pos).view(-1).float()
        neg = torch.cat(neg).view(-1).float()

        # actual data to comute stats at the end
        tp, tn, fn, fp = pos.sum(), neg.sum(), (1-pos).sum(), (1-neg).sum()

        tpr, tnr, balacc, harmacc, f1 = tpr_tnr_balacc_harmacc_f1(tp, tn, fp, fn)
        # actual interesting metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_balacc', balacc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.cp == "undef":
            enrich = enrich_no_cp
            generate_negative = generate_negative_no_cp
        elif self.cp == "bad":
            enrich = enrich_no_cp
            generate_negative = partial(generate_negative_no_cp, cp_undefined=False)
        else:
            enrich = enrich_cp
            generate_negative = generate_negative_cp
    
        # @lightning method
        a,b,c,d = batch

        # compute the embeddings
        a = self.emb(a)
        b = self.emb(b)
        c = self.emb(c)
        d = self.emb(d)

        loss = torch.tensor(0, device=a.device, dtype=float)
        pos, neg = [], []

        # positive example, target is 1
        for a_, b_, c_, d_ in enrich(a, b, c, d):
            is_analogy = self.clf(a_, b_, c_, d_)

            expected = torch.ones(is_analogy.size(), device=is_analogy.device)
            loss += self.criterion(is_analogy, expected)
            pos.append(is_analogy >= 0.5)

            # negative example, target is 0
            for a__, b__, c__, d__ in generate_negative(a_, b_, c_, d_):
                is_analogy = self.clf(a__, b__, c__, d__)

                expected = torch.zeros(is_analogy.size(), device=is_analogy.device)
                loss += self.criterion(is_analogy, expected)
                neg.append(is_analogy < 0.5)

        pos = torch.cat(pos).view(-1).float()
        neg = torch.cat(neg).view(-1).float()

        # actual data to comute stats at the end
        tp, tn, fn, fp = pos.sum(), neg.sum(), (1-pos).sum(), (1-neg).sum()

        tpr, tnr, balacc, harmacc, f1 = tpr_tnr_balacc_harmacc_f1(tp, tn, fp, fn)
        # actual interesting metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_balacc', balacc, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        if self.cp == "undef":
            #global enrich, generate_negative, n_pos_n_neg
            enrich = enrich_no_cp
            generate_negative = generate_negative_no_cp
        elif self.cp == "bad":
            enrich = enrich_no_cp
            generate_negative = partial(generate_negative_no_cp, cp_undefined=False)
        else:
            enrich = enrich_cp
            generate_negative = generate_negative_cp

        # @lightning method
        a,b,c,d = batch

        # compute the embeddings
        a_e = self.emb(a)
        b_e = self.emb(b)
        c_e = self.emb(c)
        d_e = self.emb(d)

        loss = torch.tensor(0, device=a.device, dtype=float)
        pos, neg = [], []

        def log_fails(permutation, is_analogy, positive=True):
            a, b, c, d = permutation
            fail_mask = is_analogy < 0.5 if positive else is_analogy >= 0.5

            for i, fail in enumerate(fail_mask):
                if fail:
                    self.logger[1].experiment.log_metrics({
                        "type": "False Negative" if positive else "False Positive",
                        "A": self.encoder.decode(a[i], pad_char=''),
                        "B": self.encoder.decode(b[i], pad_char=''),
                        "C": self.encoder.decode(c[i], pad_char=''),
                        "D": self.encoder.decode(d[i], pad_char=''),
                    })

        # positive example, target is 1
        for (a_, a_e_), (b_, b_e_), (c_, c_e_), (d_, d_e_) in enrich((a, a_e), (b, b_e), (c, c_e), (d, d_e)):
            is_analogy = self.clf(a_e_, b_e_, c_e_, d_e_)

            expected = torch.ones_like(is_analogy)
            loss += self.criterion(is_analogy, expected)
            pos.append(is_analogy >= 0.5)
            #log_fails((a_, b_, c_, d_), is_analogy, positive=True)

            # negative example, target is 0
            for (a__, a_e__), (b__, b_e__), (c__, c_e__), (d__, d_e__) in generate_negative((a_, a_e_), (b_, b_e_), (c_, c_e_), (d_, d_e_)):
                is_analogy = self.clf(a_e__, b_e__, c_e__, d_e__)

                if self.drop_fake_negative:
                    m: torch.Tensor = mask_valid(a__, b__, c__, d__)
                    
                    is_analogy = is_analogy[~m]
                    expected = torch.zeros_like(is_analogy)
                    loss += self.criterion(is_analogy, expected)
                    neg.append(is_analogy < 0.5)
                    #log_fails((a__[~m], b__[~m], c__[~m], d__[~m]), is_analogy, positive=False)
                else:
                    expected = torch.zeros_like(is_analogy)
                    loss += self.criterion(is_analogy, expected)
                    neg.append(is_analogy < 0.5)
                    #log_fails((a__, b__, c__, d__), is_analogy, positive=False)

        pos = torch.cat(pos).view(-1).float()
        neg = torch.cat(neg).view(-1).float()

        # actual data to comute stats at the end
        tp, tn, fn, fp = pos.sum(), neg.sum(), (1-pos).sum(), (1-neg).sum()
        self.logger[1].log_metrics({
                        "true_positive": tp,
                        "true_negative": tn,
                        "false_positive": fp,
                        "false_negative": fn
                    })
        
        tpr, tnr, balacc, harmacc, f1 = tpr_tnr_balacc_harmacc_f1(tp, tn, fp, fn)
        # actual interesting metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_balacc', balacc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # as averaging harmonc accuracies of the batches is not equivalent to the harmonic accuracy of the whole, the
        #    final averaged value of 'test_harmacc_approx' will only be an approximation of the harmonic accuracy.
        self.log('test_harmacc_approx', harmacc, on_step=False, on_epoch=True, prog_bar=True, logger=False, sync_dist=True)

def train_one(args, train_cp, train_loader, val_loader, encoder):
    # --- Define models ---
    char_emb_size = 64

    nn = ClfLightning(char_emb_size=char_emb_size, encoder=encoder, filters=args.filters, drop_fake_negative=args.drop_fake_negative, cp=train_cp)
    
    # --- Train model ---
    cp_tag = "cp-" + train_cp if train_cp else "cp"
    expe_name = f"clf-cp-no-cp/{args.dataset}/{args.language}/{cp_tag}"
    tb_logger = pl.loggers.TensorBoardLogger("logs/", expe_name, version=args.version)
    trainer_kwargs = {"strategy" if USE_STRATEGY else "plugins": DDPSpawnPlugin(find_unused_parameters=False) if USE_SPAWN # causes dataloader issues
                else DDPPlugin(find_unused_parameters=False)}
    checkpoint_callback=ModelCheckpoint(
        filename=f"clf-{cp_tag}-{args.dataset}-{args.language}-b{args.batch_size}-{{epoch:02d}}",
        monitor="val_balacc", mode="max", save_top_k=0, save_last=True)
    trainer = pl.Trainer.from_argparse_args(args,
        callbacks=[
            #EarlyStopping(monitor="val_loss"),
            checkpoint_callback,
        ],
        logger = tb_logger,
        **trainer_kwargs
    )

    seed_everything(MODEL_RANDOM_SEEDS[args.version], workers=True)
    trainer.fit(nn, train_loader, val_loader)

    #logger.info(f"best model path: {checkpoint_callback.best_model_path} (validation balanced accuracy: {checkpoint_callback.best_model_score:.3f})")
    logger.info(f"last model path: {checkpoint_callback.last_model_path}")

    return nn, checkpoint_callback.last_model_path

def test_one(args, train_cp, test_cp, test_loader, encoder, nn):
    #char_emb_size = 64
    #nn = ClfLightning(char_emb_size=char_emb_size, encoder=encoder, filters=args.filters, drop_fake_negative=args.drop_fake_negative, cp=test_cp)
    #nn.load_from_checkpoint(ckpt_path)
    nn.cp = test_cp

    cp_tag = ("cp-" + train_cp if train_cp else "cp")
    test_cp_tag = ("cp-" + test_cp if test_cp else "cp")
    expe_name = f"clf-cp-no-cp/{args.dataset}/{args.language}/{cp_tag}/{test_cp_tag}"
    tb_logger = pl.loggers.TensorBoardLogger("logs/", expe_name, version=args.version)

    cpnocp_logger = CPnoCPLogger()
    trainer_kwargs = {"strategy" if USE_STRATEGY else "plugins": DDPSpawnPlugin(find_unused_parameters=False) if USE_SPAWN # causes dataloader issues
                else DDPPlugin(find_unused_parameters=False)}
    trainer = pl.Trainer.from_argparse_args(args,
        logger = [tb_logger, cpnocp_logger],
        **trainer_kwargs
    )
    trainer.test(nn, dataloaders=test_loader)
    tpr, tnr, balacc, harmacc, f1 = cpnocp_logger.tpr, cpnocp_logger.tnr, cpnocp_logger.balacc, cpnocp_logger.harmacc, cpnocp_logger.f1

    return tpr, tnr, balacc, harmacc, f1

@rank_zero_only
def conf_matrices(arrays, args):
    os.makedirs(f"results/clf-cp-no-cp/{args.dataset}/{args.language}/", exist_ok=True)
    labels = ["cp", "cp undef", "cp bad"]

    # create an empty df and fill it
    indices = pd.MultiIndex.from_product((arrays.keys(), labels, labels), names=('Measure', 'Train setup', 'Test setup'))
    df = pd.DataFrame(0, index=indices, columns=('value',), dtype=float)
    for value in arrays.keys():
        for i, train_values in enumerate(arrays[value]):
            for j, test_value in enumerate(train_values):
                df["value"][(value,labels[i],labels[j])] = test_value

    df.to_csv(f"results/clf-cp-no-cp/{args.dataset}/{args.language}/{args.version}.csv")
    df = df.reset_index()

    import matplotlib.pyplot as plt, seaborn as sns
    def draw_heatmap(*args, **kwargs):
        data = kwargs.pop('data')
        d = data.pivot(index=args[1], columns=args[0], values=args[2])
        d = d.sort_index(key=lambda x: x.map(lambda y: labels.index(y)))
        d = d.sort_index(axis=1, key=lambda x: x.map(lambda y: labels.index(y)))
        sns.heatmap(d, square = True, annot=True, cmap="RdYlGn", fmt=".3f", cbar=False, **kwargs)

    fg = sns.FacetGrid(df, col='Measure')
    fg.map_dataframe(draw_heatmap, 'Test setup', 'Train setup', 'value')
    fg.set_titles("{col_name}")
    fg.figure.subplots_adjust(top=0.8)
    fg.figure.suptitle(f"Performance of the model on '{args.dataset} - {args.language.capitalize()} - V{args.version}' {'(post filtering of negative samples applied' if args.drop_fake_negative else ''}")
    plt.savefig(f"results/clf-cp-no-cp/{args.dataset}/{args.language}/{args.version}")
    torch.save(arrays, f"results/clf-cp-no-cp/{args.dataset}/{args.language}/{args.version}.pkl")

@rank_zero_only
def conf_matrices_all_v(arrays, args):
    os.makedirs(f"results/clf-cp-no-cp/{args.dataset}/{args.language}/", exist_ok=True)
    labels = ["cp", "cp undef", "cp bad"]

    # create an empty df and fill it
    indices = pd.MultiIndex.from_product((arrays[0].keys(), labels, labels, arrays.keys()), names=('Measure', 'Train setup', 'Test setup', 'v'))
    df_ = pd.DataFrame(0, index=indices, columns=('value',), dtype=float)
    for v, array in arrays.items():
        for value in array.keys():
            for i, train_values in enumerate(array[value]):
                for j, test_value in enumerate(train_values):
                    df_["value"][(value,labels[i],labels[j],v)] = test_value

    #df_ = df_.reset_index()
    df_.to_csv(f"results/clf-cp-no-cp/{args.dataset}/{args.language}/all-v.csv")
    gb = df_.groupby(['Measure', 'Train setup', 'Test setup'])
    df = gb.mean()
    df["label"] = df["value"].apply(lambda x: f"{x:.3f}") + '\n±' + gb.std()["value"].apply(lambda x: f"{x:.3f}")
    df = df.reset_index()

    import matplotlib.pyplot as plt, seaborn as sns
    def draw_heatmap(*args, **kwargs):
        data = kwargs.pop('data')
        d = data.pivot(index=args[1], columns=args[0], values=args[2])
        d = d.sort_index(key=lambda x: x.map(lambda y: labels.index(y)))
        d = d.sort_index(axis=1, key=lambda x: x.map(lambda y: labels.index(y)))
        labels_df: pd.DataFrame = data.pivot(index=args[1], columns=args[0], values='label')
        labels_df = labels_df.sort_index(key=lambda x: x.map(lambda y: labels.index(y)))
        labels_df = labels_df.sort_index(axis=1, level='test setup',key=lambda x: x.map(lambda y: labels.index(y)))
        sns.heatmap(d, annot=labels_df.values, fmt="s", cmap="RdYlGn", cbar=False, square = True, **kwargs)

    fg = sns.FacetGrid(df, col='Measure', height=5)
    fg.map_dataframe(draw_heatmap, 'Test setup', 'Train setup', 'value')
    fg.set_titles("{col_name}")
    fg.figure.subplots_adjust(top=0.8)
    fg.figure.suptitle(f"Performance of the model on '{args.dataset} - {args.language.capitalize()}' {'(post filtering of negative samples applied' if args.drop_fake_negative else ''}")
    plt.savefig(f"results/clf-cp-no-cp/{args.dataset}/{args.language}/all-v")
    torch.save(arrays, f"results/clf-cp-no-cp/{args.dataset}/{args.language}/all-v.pkl")

def main(args):
    logger.warning("Determinism (--deterministic True) does not guarentee reproductible results when changing the number of processes.")

    train_loader, val_loader, test_loader, encoder = prepare_data(
        args.dataset, args.language, args.nb_analogies_train, args.nb_analogies_val, args.nb_analogies_test,
        args.batch_size, args.force_rebuild, split_seed=MODEL_RANDOM_SEEDS[args.version])

    # --- Define models ---
    cp_modes = ["", "undef", "bad"]
    results = {"TPR": [], "TNR": [], "bal. acc.": [], "harm. acc.": [], "F1": []}
    for train_cp in cp_modes:
        logger.info(f"Training seed {args.version}, {train_cp}")
        nn, model_path = train_one(args, train_cp, train_loader, val_loader, encoder)
        results_this_train_cp = {"TPR": [], "TNR": [], "bal. acc.": [], "harm. acc.": [], "F1": []}
        for test_cp in cp_modes:
            logger.info(f"Testing seed {args.version}, {train_cp} on {test_cp}")
            with torch.no_grad():
                tpr, tnr, balacc, harmacc, f1 = test_one(args, train_cp, test_cp, test_loader, encoder, nn)
                results_this_train_cp["TPR"].append(tpr)
                results_this_train_cp["TNR"].append(tnr)
                results_this_train_cp["bal. acc."].append(balacc)
                results_this_train_cp["harm. acc."].append(harmacc)
                results_this_train_cp["F1"].append(f1)
        results["TPR"].append(results_this_train_cp["TPR"])
        results["TNR"].append(results_this_train_cp["TNR"])
        results["bal. acc."].append(results_this_train_cp["bal. acc."])
        results["harm. acc."].append(results_this_train_cp["harm. acc."])
        results["F1"].append(results_this_train_cp["F1"])

    # --- Draw ---
    conf_matrices(results, args)
    return results

def main_all_v(args):
    results = dict()
    for i in range(len(MODEL_RANDOM_SEEDS)):
        args.version = i
        logger.info(f"Processing seed {i} ({MODEL_RANDOM_SEEDS[i]})")
        if os.path.exists(f"results/clf-cp-no-cp/{args.dataset}/{args.language}/{args.version}.pkl"):
            results[i] = torch.load(f"results/clf-cp-no-cp/{args.dataset}/{args.language}/{args.version}.pkl", map_location='cpu')
        else:
            results[i] = main(args)

    # --- Draw ---
    conf_matrices_all_v(results, args)
    
if __name__ == '__main__':
    from argparse import ArgumentParser

    # argument parsing
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    model_parser = parser.add_argument_group("Model arguments")
    model_parser.add_argument('--filters', '-f', type=int, default=128, help='The number of filters of the classification model.')

    dataset_parser = parser.add_argument_group("Dataset arguments")
    dataset_parser.add_argument('--dataset', '-d', type=str, default="2016", help='The language to train the model on.', choices=["2016", "2019"])
    dataset_parser.add_argument('--language', '-l', type=str, default="arabic", help='The language to train the model on.', choices=SIG2019_HIGH+SIG2016_LANGUAGES)
    dataset_parser.add_argument('--force-rebuild', help='Force the re-building of the dataset file.',  action='store_true')
    dataset_parser.add_argument('--nb-analogies-train', '-n', type=int, default=50000, help='The maximum number of analogies (before augmentation) we train the model on.')
    dataset_parser.add_argument('--nb-analogies-val', '-v', type=int, default=500, help='The maximum number of analogies (before augmentation) we validate the model on.')
    dataset_parser.add_argument('--nb-analogies-test', '-t', type=int, default=50000, help='The maximum number of analogies (before augmentation) we test the model on.')
    dataset_parser.add_argument('--batch-size', '-b', type=int, default=32, help='Batch size.')
    dataset_parser.add_argument('--version', '-V', type=int, default=0, help='The experiment version, also corresponding to the random seed to use.')
    dataset_parser.add_argument('--drop-fake-negative', help='Ignore negetive permutations that end up being a:a::b:b or a:b::a:b.', action='store_true')

    args = parser.parse_args()

    #main(args)
    main_all_v(args)