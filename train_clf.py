from functools import partial
import logging
from packaging import version

from utils.logger import append_pkl, to_csv
logger = logging.getLogger("")#__name__)
logger.setLevel(logging.INFO)

# configure logging at the root level of lightning
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)
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

import config
from analogy_clf import AnalogyClassification
from cnn_embeddings import CNNEmbedding
from siganalogies import SIG2016_LANGUAGES, SIG2019_HIGH, enrich, generate_negative, n_pos_n_neg, enrich_no_cp, generate_negative_no_cp, n_pos_n_neg_no_cp
from utils import prepare_data, tpr_tnr_balacc_harmacc_f1, mask_valid

import os
os.environ['PYTHONHASHSEED'] = str(42)
seed_everything(42, workers=True)

MODEL_RANDOM_SEEDS = [42, 8564851, 706303, 248, 8994204, 7332146, 800, 3347863, 1402754, 7938707]

class ClfLightning(pl.LightningModule):
    def __init__(self, char_emb_size, encoder, filters = 128, drop_fake_negative=False):
        super().__init__()
        self.save_hyperparameters()
        self.emb = CNNEmbedding(voc_size=len(encoder.id_to_char), char_emb_size=char_emb_size)
        self.clf = AnalogyClassification(emb_size=self.emb.get_emb_size(), filters = filters)
        self.encoder = encoder
        
        self.criterion = nn.BCELoss()

        self.drop_fake_negative = drop_fake_negative

        self.save_path = ""
        self.common_save_path = ""
        self.extra_info = dict()

    def configure_optimizers(self):
        # @lightning method
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
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
        # @lightning method
        a,b,c,d = batch

        # compute the embeddings
        a_e = self.emb(a)
        b_e = self.emb(b)
        c_e = self.emb(c)
        d_e = self.emb(d)

        loss = torch.tensor(0, device=a.device, dtype=float)
        pos, neg = [], []

        fails = []

        def log_fails(permutation, is_analogy, positive=True):
            a, b, c, d = permutation
            fail_mask = is_analogy < 0.5 if positive else is_analogy >= 0.5

            for i, fail in enumerate(fail_mask):
                if fail:
                    fails.append({
                        "type": "False Negative" if positive else "False Positive",
                        "A": self.encoder.decode(a[i], pad_char=''),
                        "B": self.encoder.decode(b[i], pad_char=''),
                        "C": self.encoder.decode(c[i], pad_char=''),
                        "D": self.encoder.decode(d[i], pad_char=''),
                    })
                    # self.logger[1].experiment.log_metrics({
                    #     "type": "False Negative" if positive else "False Positive",
                    #     "A": self.encoder.decode(a[i], pad_char=''),
                    #     "B": self.encoder.decode(b[i], pad_char=''),
                    #     "C": self.encoder.decode(c[i], pad_char=''),
                    #     "D": self.encoder.decode(d[i], pad_char=''),
                    # })

        # positive example, target is 1
        for (a_, a_e_), (b_, b_e_), (c_, c_e_), (d_, d_e_) in enrich((a, a_e), (b, b_e), (c, c_e), (d, d_e)):
            is_analogy = self.clf(a_e_, b_e_, c_e_, d_e_)

            expected = torch.ones_like(is_analogy)
            loss += self.criterion(is_analogy, expected)
            pos.append(is_analogy >= 0.5)
            log_fails((a_, b_, c_, d_), is_analogy, positive=True)

            # negative example, target is 0
            for (a__, a_e__), (b__, b_e__), (c__, c_e__), (d__, d_e__) in generate_negative((a_, a_e_), (b_, b_e_), (c_, c_e_), (d_, d_e_)):
                is_analogy = self.clf(a_e__, b_e__, c_e__, d_e__)

                if self.drop_fake_negative:
                    m: torch.Tensor = mask_valid(a__, b__, c__, d__)
                    
                    is_analogy = is_analogy[~m]
                    expected = torch.zeros_like(is_analogy)
                    loss += self.criterion(is_analogy, expected)
                    neg.append(is_analogy < 0.5)
                    log_fails((a__[~m], b__[~m], c__[~m], d__[~m]), is_analogy, positive=False)
                else:
                    expected = torch.zeros_like(is_analogy)
                    loss += self.criterion(is_analogy, expected)
                    neg.append(is_analogy < 0.5)
                    log_fails((a__, b__, c__, d__), is_analogy, positive=False)

        pos = torch.cat(pos).view(-1).float()
        neg = torch.cat(neg).view(-1).float()

        # actual data to comute stats at the end
        tp, tn, fn, fp = pos.sum(), neg.sum(), (1-pos).sum(), (1-neg).sum()
        self.log('true_positive', tp, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True) # true positive
        self.log('true_negative', tn, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True) # true negative
        self.log('false_positive', fp, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True) # false positive
        self.log('false_negative', fn, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True) # false negative
        
        tpr, tnr, balacc, harmacc, f1 = tpr_tnr_balacc_harmacc_f1(tp, tn, fp, fn)
        # actual interesting metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_balacc', balacc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # as averaging harmonc accuracies of the batches is not equivalent to the harmonic accuracy of the whole, the
        #    final averaged value of 'test_harmacc_approx' will only be an approximation of the harmonic accuracy.
        self.log('test_harmacc_approx', harmacc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


        return {
            "scores": {"tp": tp, "tn": tn, "fn": fn, "fp": fp},
            "fails": fails
        }

    def test_epoch_end(self, outputs):
        gathered = self.all_gather(outputs)

        # When logging only on rank 0, don't forget to add ``rank_zero_only=True`` to avoid deadlocks on synchronization.
        if self.trainer.is_global_zero:
            fails = []
            for gathered_ in gathered:
                fails.extend(gathered_["fails"])

            tp = sum(gathered_["scores"].get("tp", 0) for gathered_ in gathered).sum()
            tn = sum(gathered_["scores"].get("tn", 0) for gathered_ in gathered).sum()
            fp = sum(gathered_["scores"].get("fp", 0) for gathered_ in gathered).sum()
            fn = sum(gathered_["scores"].get("fn", 0) for gathered_ in gathered).sum()
            
            tpr, tnr, balacc, harmacc, f1 = tpr_tnr_balacc_harmacc_f1(tp, tn, fp, fn)


            row = {"balacc": balacc.item(), "harmacc": harmacc.item(), "TPR": tpr.item(), "TNR": tnr.item(), "F1": f1.item(), **self.extra_info}
            print(row)
            append_pkl(self.common_save_path, row)
            #print(fails)
            to_csv(os.path.join(self.save_path, "summary.csv"), row)
            to_csv(os.path.join(self.save_path, "fails.csv"), fails)
            #self.log("my_reduced_metric", mean, rank_zero_only=True)

def main(args):
    cp_tag = "-cp-" + args.cp if args.cp else ""
    expe_name = f"clf{cp_tag}/{args.dataset}/{args.language}"
    if args.skip and os.path.exists(f"logs/{expe_name}/version_{args.version}/summary.csv"):
        print(f"logs/{expe_name}/version_{args.version}/summary.csv exists, skip")
        return


    global enrich, generate_negative, n_pos_n_neg
    if args.cp == "undef":
        #global enrich, generate_negative, n_pos_n_neg
        enrich = enrich_no_cp
        generate_negative = generate_negative_no_cp
        n_pos_n_neg = n_pos_n_neg_no_cp
    elif args.cp == "bad":
        enrich = enrich_no_cp
        generate_negative = partial(generate_negative_no_cp, cp_undefined=False)
        n_pos_n_neg = partial(n_pos_n_neg_no_cp, cp_undefined=False)
    use_spawn = False
    use_strategy = version.parse(pl.__version__) >= version.parse("1.5.0")

    logger.warning("Determinism (--deterministic True) does not guarentee reproductible results when changing the number of processes.")

    from pytorch_lightning.plugins import DDPSpawnPlugin, DDPPlugin, DDP2Plugin
    train_loader, val_loader, test_loader, encoder = prepare_data(
        args.dataset, args.language, args.nb_analogies_train, args.nb_analogies_val, args.nb_analogies_test,
        args.batch_size, args.force_rebuild, split_seed=MODEL_RANDOM_SEEDS[args.version])

    # --- Define models ---
    char_emb_size = 64

    nn = ClfLightning(char_emb_size=char_emb_size, encoder=encoder, filters=args.filters, drop_fake_negative=args.drop_fake_negative)

    # --- Train model ---
    tb_logger = pl.loggers.TensorBoardLogger("logs/no-train/" if args.no_train else "logs/", expe_name, version=args.version)
    trainer_kwargs = {"strategy" if use_strategy else "plugins": DDPSpawnPlugin(find_unused_parameters=False) if use_spawn # causes dataloader issues
                else DDPPlugin(find_unused_parameters=False)}
    if not args.no_train:
        checkpoint_callback=ModelCheckpoint(
            filename=f"clf{cp_tag}-{args.dataset}-{args.language}-b{args.batch_size}-{{epoch:02d}}",
            monitor="val_balacc", mode="max", save_top_k=1)
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

        logger.info(f"best model path: {checkpoint_callback.best_model_path} (validation balanced accuracy: {checkpoint_callback.best_model_score:.3f})")
    
    with torch.no_grad():
        trainer = pl.Trainer.from_argparse_args(args,
            logger = tb_logger,
            **trainer_kwargs
        )
        save_dir="logs/no-train" if args.no_train else "logs"
        nn.save_path = os.path.join(save_dir, expe_name, f"version_{args.version}")
        nn.extra_info = {
            "best_model": args.ckpt or checkpoint_callback.best_model_path,
            "seed": MODEL_RANDOM_SEEDS[args.version],
            "seed_id": args.version,
            "lang": args.language,
            "dataset": args.dataset,
            "variant": {"undef": "central permutation undefined", "bad": "central permutation bad" , "": "central permutation"}[args.cp]
        }
        nn.common_save_path = "results/ret.pkl"
        if args.ckpt:
            trainer.test(nn, dataloaders=test_loader, ckpt_path=args.ckpt)
        else:
            trainer.test(nn, dataloaders=test_loader)#, ckpt_path=checkpoint_callback.best_model_path)


def add_argparse_args(parser):
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
    dataset_parser.add_argument('--ckpt', type=str, help='Checkpoint.', default="")
    dataset_parser.add_argument('--no-train', help='If used no training will be done. To use with --ckpt [CHECKPOINT PATH] to re-test a particular model with no training.', action='store_true')
    dataset_parser.add_argument('--cp', type=str, default="", help='If used, will switch data augmentation to "no central permutation".', choices=["", "bad", "undef"])
    dataset_parser.add_argument('--skip', help='Skip if such a model has been trained already.', action='store_true')

    return parser, dataset_parser

if __name__ == '__main__':
    from argparse import ArgumentParser

    # argument parsing
    parser = ArgumentParser()
    parser, dataset_parser = add_argparse_args(parser)

    args = parser.parse_args()

    assert args.ckpt or not args.no_train

    #try:
    main(args)
    #except RuntimeError or ProcessLookupError as e:
    #    import traceback
    #    logger.error(f"Caught {e}:")
    #    traceback.print_tb(e.__traceback__)
