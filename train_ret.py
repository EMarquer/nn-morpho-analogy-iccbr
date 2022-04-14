import logging
from math import ceil
from packaging import version
from sklearn import datasets
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
from torch.nn.functional import mse_loss
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, BaseFinetuning

import config
from analogy_reg import AnalogyRegression
from analogy_reg_univ import AnalogyRegression as AnalogyRegressionCNN
from cnn_embeddings import CNNEmbedding
from siganalogies import SIG2016_LANGUAGES, SIG2019_HIGH, enrich, generate_negative, n_pos_n_neg
from utils import prepare_data, precision_sak_ap_rr, embeddings_voc

from utils.logger import append_pkl, to_csv
from utils.utils import elapsed_timer

import os
os.environ['PYTHONHASHSEED'] = str(42)
os.environ['PYTORCH_CUDA_ALLOC_CONF']="max_split_size_mb:128"
seed_everything(42, workers=True)

torch.cuda.memory.set_per_process_memory_fraction(0.9)

MODEL_RANDOM_SEEDS = [42, 8564851, 706303, 248, 8994204, 7332146, 800, 3347863, 1402754, 7938707]


class RetLightning(pl.LightningModule):
    def __init__(self, char_emb_size, encoder, filters = 128, criterion: str="cosine embedding loss", variant="classical", freeze_emb=False):
        super().__init__()
        self.save_hyperparameters()
        self.emb = CNNEmbedding(voc_size=len(encoder.id_to_char), char_emb_size=char_emb_size)
        self.variant = variant
        if self.variant == "univ":
            self.reg = AnalogyRegressionCNN(emb_size=self.emb.get_emb_size(), filters = filters, mode="ab!=ac")
        else:
            self.reg = AnalogyRegression(emb_size=self.emb.get_emb_size(), filters = filters, mode="ab!=ac")
        self.encoder = encoder
        
        self.criterion = criterion
        self.voc = None
        assert self.criterion in {"cosine embedding loss", "relative shuffle", "relative all", "all"}
        if self.criterion == "cosine embedding loss" or self.criterion == "all":
            self.cosine_embedding_loss = nn.CosineEmbeddingLoss()

        self.save_path = ""
        self.common_save_path = ""
        self.extra_info = dict()
        self.freeze_emb = freeze_emb

    def configure_optimizers(self):
        # @lightning method
        if self.freeze_emb:
            optimizer = torch.optim.Adam([
                {"params": self.reg.parameters(), "lr": 1e-3}])
        else:
            optimizer = torch.optim.Adam([
                {"params": self.emb.parameters(), "lr": 1e-5},
                {"params": self.reg.parameters(), "lr": 1e-3}])
        return optimizer

    def loss_fn(self, a, b, c, d, d_pred):
        if self.criterion == "cosine embedding loss":
            return self.cosine_embedding_loss(
                    torch.cat([d_pred]*4, dim=0),
                    torch.cat([d,a,b,c], dim=0),
                    torch.cat([torch.ones(a.size(0)), -torch.ones(a.size(0) * 3)]).to(a.device))

        elif self.criterion == "relative shuffle":
            good = mse_loss(d, d_pred)
            bad = mse_loss(d[torch.randperm(d.size(0))], d_pred)

            return (good + 1) / (bad + 1)

        elif self.criterion == "relative all":
            return (1 + mse_loss(d_pred, d) * 6) / (1 +
                mse_loss(a,b) +
                mse_loss(a,c) +
                mse_loss(a,d) +
                mse_loss(b,c) +
                mse_loss(b,d) +
                mse_loss(c,d))
        
        else:
            good = mse_loss(d, d_pred)
            bad = mse_loss(d[torch.randperm(d.size(0))], d_pred)
            return (
                self.cosine_embedding_loss(
                    torch.cat([d_pred]*4, dim=0),
                    torch.cat([d,a,b,c], dim=0),
                    torch.cat([torch.ones(a.size(0)), -torch.ones(a.size(0) * 3)]).to(a.device)) 
                    + (good + 1) / (bad + 1)
                    + ((1 + mse_loss(d_pred, d) * 6) / (1 +
                mse_loss(a,b) +
                mse_loss(a,c) +
                mse_loss(a,d) +
                mse_loss(b,c) +
                mse_loss(b,d) +
                mse_loss(c,d))))

    def training_step(self, batch, batch_idx):
        # @lightning method
        a,b,c,d = batch

        # compute the embeddings
        a = self.emb(a)
        b = self.emb(b)
        c = self.emb(c)
        d = self.emb(d)

        loss = torch.tensor(0, device=a.device, dtype=float)

        # positive examples
        for a_, b_, c_, d_ in enrich(a, b, c, d):
            d_pred = self.reg(a_, b_, c_)

            loss += self.loss_fn(a_, b_, c_, d_, d_pred)

        # actual interesting metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
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
        
        # positive examples
        for a_, b_, c_, d_ in enrich(a, b, c, d):
            d_pred = self.reg(a_, b_, c_)

            loss += self.loss_fn(a_, b_, c_, d_, d_pred)
        
        # actual interesting metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

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
        scores = []
        fails = []
        
        def log_fails(permutation, d_e_, d_pred):
            a, b, c, d = permutation
            
            #with elapsed_timer() as t:
            p, sak, r, rr, pred_w, tgt_w = precision_sak_ap_rr(d_pred, d_e_, self.voc, k=[3,5,10])

            mask = p < 1
            indices = torch.arange(a.size(0), device=p.device)[mask]
            for i in indices:
                fails.append({
                        "A": self.encoder.decode(a[i], pad_char=''),
                        "B": self.encoder.decode(b[i], pad_char=''),
                        "C": self.encoder.decode(c[i], pad_char=''),
                        "actual D": self.encoder.decode(d[i], pad_char=''),
                        "predicted D": pred_w[i],
                        "target D": tgt_w[i],
                    })
            scores.append({
                'precision': p,
                'success@3': sak[0],
                'success@5': sak[1],
                'success@10': sak[2],
                'rr': rr
            })

        # positive example, target is 1
        for (a_, a_e_), (b_, b_e_), (c_, c_e_), (d_, d_e_) in enrich((a, a_e), (b, b_e), (c, c_e), (d, d_e)):
            d_pred = self.reg(a_e_, b_e_, c_e_)
            loss += self.loss_fn(a_e_, b_e_, c_e_, d_e_, d_pred)
            log_fails((a_, b_, c_, d_), d_e_, d_pred)

        return {
            "scores": scores,
            "fails": fails
        }

    def test_epoch_end(self, outputs):
        gathered = self.all_gather(outputs)

        # When logging only on rank 0, don't forget to add ``rank_zero_only=True`` to avoid deadlocks on synchronization.
        if self.trainer.is_global_zero:
            scores = []
            fails = []
            for gathered_ in gathered:
                scores.extend(gathered_["scores"])
                fails.extend(gathered_["fails"])

            m_precision = torch.mean(torch.cat([score_dict["precision"] for score_dict in scores], dim=-1))
            m_sak3 = torch.mean(torch.cat([score_dict["success@3"] for score_dict in scores], dim=-1))
            m_sak5 = torch.mean(torch.cat([score_dict["success@5"] for score_dict in scores], dim=-1))
            m_sak10 = torch.mean(torch.cat([score_dict["success@10"] for score_dict in scores], dim=-1))
            m_rr = torch.mean(torch.cat([score_dict["rr"] for score_dict in scores], dim=-1))

            row = {"precision": m_precision.item(), "success@3": m_sak3.item(), "success@5": m_sak5.item(), "success@10": m_sak10.item(), "mrr": m_rr.item(), **self.extra_info}
            print(row)
            append_pkl(self.common_save_path, row)
            #print(fails)
            to_csv(os.path.join(self.save_path, "summary.csv"), row)
            to_csv(os.path.join(self.save_path, "fails.csv"), fails)
            #self.log("my_reduced_metric", mean, rank_zero_only=True)

class EmbFreezeUnfreeze(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10, lr=1e-5):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self.lr=lr

    def freeze_before_training(self, pl_module: RetLightning):
        # freeze any module you want
        # Here, we are freezing `feature_extractor`
        self.freeze(pl_module.emb)

    def finetune_function(self, pl_module: RetLightning, current_epoch, optimizer, optimizer_idx):
        # When `current_epoch` is 10, feature_extractor will start training.
        if current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=pl_module.emb,
                optimizer=optimizer,
                train_bn=True,
                lr=self.lr
            )

def main(args):
    expe_name = f"ret/{args.dataset}/{args.language}/{args.criterion.replace(' ', '_')}{'' if args.transfer else '-fromscratch'}"
    if args.skip and os.path.exists(f"logs/{expe_name}/version_{args.version}/summary.csv"):
        print(f"logs/{expe_name}/version_{args.version}/summary.csv exists, skip")
        return


    use_spawn = False
    use_strategy = version.parse(pl.__version__) >= version.parse("1.5.0")

    logger.warning("Determinism (--deterministic True) does not guarantee reproducible results when changing the number of processes.")

    from pytorch_lightning.plugins import DDPSpawnPlugin, DDPPlugin, DDP2Plugin
    train_loader, val_loader, test_loader, encoder = prepare_data(
        args.dataset, args.language, args.nb_analogies_train, args.nb_analogies_val, args.nb_analogies_test,
        args.batch_size, args.force_rebuild, split_seed=MODEL_RANDOM_SEEDS[args.version])

    if args.max_epochs is not None:
        args.max_epochs = args.max_epochs * ceil(args.nb_analogies_train / len(train_loader.dataset))

    # --- Define models ---
    char_emb_size = 64

    seed_everything(MODEL_RANDOM_SEEDS[args.version], workers=True)
    nn = RetLightning(char_emb_size=char_emb_size, encoder=encoder, filters=args.filters, variant=args.variant, criterion=args.criterion)

    #transfer_name = ""
    emb_loaded=False
    if args.transfer:
        try:
            state_dict = torch.load(args.transfer, map_location="cpu")["state_dict"]
            state_dict_emb = {k[len("emb."):]: v for k, v in state_dict.items() if k.startswith("emb.")}
            nn.emb.load_state_dict(state_dict_emb)
            #transfer_name = "/transfer/"+args.transfer
            logger.warning(f"Successfully loaded embedding from {args.transfer}")
            emb_loaded=True
        except Exception: pass
        try:
            state_dict = torch.load(args.transfer, map_location="cpu")["state_dict"]
            state_dict_reg = {k[len("reg."):]: v for k, v in state_dict.items() if k.startswith("reg.")}
            nn.reg.load_state_dict(state_dict_reg)
            #transfer_name = "/transfer/"+args.transfer
            logger.warning(f"Successfully loaded regression network from {args.transfer}")
        except Exception: pass

    # --- Train model ---
    #expe_name = f"ret/{args.dataset}/{args.language}{transfer_name}/{args.criterion.replace(' ', '_')}"
    tb_logger = pl.loggers.TensorBoardLogger('logs/', expe_name, version=args.version)
    checkpoint_callback=ModelCheckpoint(
        filename=f"ret-{args.dataset}-{args.language}-b{args.batch_size}-{{epoch:02d}}",
        monitor="val_loss", mode="min", save_top_k=1)
    
    trainer_kwargs = {"strategy" if use_strategy else "plugins": DDPSpawnPlugin(find_unused_parameters=False) if use_spawn # causes dataloader issues
            else DDPPlugin(find_unused_parameters=False)}
    
    # with emb frozen
    if args.freeze_emb and args.transfer:
        nn.freeze_emb = True
        trainer = pl.Trainer.from_argparse_args(args,
            callbacks=[
                EarlyStopping(monitor="val_loss"),
                checkpoint_callback,
                #EmbFreezeUnfreeze(unfreeze_at_epoch=10, lr=1e-5)
            ],
            logger = tb_logger,
            **trainer_kwargs
        )

        seed_everything(MODEL_RANDOM_SEEDS[args.version], workers=True)
        trainer.fit(nn, train_loader, val_loader)
        epochs_no_emb = nn.current_epoch + 1
    else:
        epochs_no_emb = 0

    # with emb unfrozen
    if args.max_epochs is not None:
        args.max_epochs = args.max_epochs - epochs_no_emb
    nn.freeze_emb = False
    trainer = pl.Trainer.from_argparse_args(args,
        callbacks=[
            EarlyStopping(monitor="val_loss"),
            checkpoint_callback,
            #EmbFreezeUnfreeze(unfreeze_at_epoch=10, lr=1e-5)
        ],
        logger = tb_logger,
        **trainer_kwargs
    )

    seed_everything(MODEL_RANDOM_SEEDS[args.version], workers=True)
    trainer.fit(nn, train_loader, val_loader)


    with torch.no_grad():
        with embeddings_voc(nn.emb, train_loader.dataset.dataset, test_loader.dataset.dataset) as voc:
            trainer = pl.Trainer.from_argparse_args(args,
            logger = None,
                **trainer_kwargs
            )
            nn.voc = voc
            nn.voc.vectors = nn.voc.vectors.to(nn.device)
            nn.save_path = os.path.join('logs', expe_name, f"version_{args.version}")
            nn.extra_info = {
                "best_model": checkpoint_callback.best_model_path,
                "seed": MODEL_RANDOM_SEEDS[args.version],
                "seed_id": args.version,
                "lang": args.language,
                "dataset": args.dataset,
                "variant": args.variant,
                "criterion": args.criterion,
                "transfer": args.transfer,
                "epochs_no_emb": epochs_no_emb,
                "epochs_total": epochs_no_emb + nn.current_epoch + 1}
            nn.common_save_path = "results/ret.pkl"
            trainer.test(nn, dataloaders=test_loader, ckpt_path=checkpoint_callback.best_model_path)
            nn.voc = None

def add_argparse_args(parser):
    # argument parsing
    parser = pl.Trainer.add_argparse_args(parser)

    model_parser = parser.add_argument_group("Model arguments")
    model_parser.add_argument('--filters', '-f', type=int, default=128, help='The number of filters of the retrival model.')

    dataset_parser = parser.add_argument_group("Dataset arguments")
    dataset_parser.add_argument('--dataset', '-d', type=str, default="2016", help='The language to train the model on.', choices=["2016", "2019"])
    dataset_parser.add_argument('--language', '-l', type=str, default="arabic", help='The language to train the model on.', choices=SIG2019_HIGH+SIG2016_LANGUAGES)
    dataset_parser.add_argument('--force-rebuild', help='Force the re-building of the dataset file.',  action='store_true')
    dataset_parser.add_argument('--no-freeze-emb', help='Freeze embedding until convergence, then unfreeze it.',  action='store_false', dest="freeze_emb")
    dataset_parser.add_argument('--nb-analogies-train', '-n', type=int, default=50000, help='The maximum number of analogies (before augmentation) we train the model on.')
    dataset_parser.add_argument('--nb-analogies-val', '-v', type=int, default=500, help='The maximum number of analogies (before augmentation) we validate the model on.')
    dataset_parser.add_argument('--nb-analogies-test', '-t', type=int, default=50000, help='The maximum number of analogies (before augmentation) we test the model on.')
    dataset_parser.add_argument('--batch-size', '-b', type=int, default=512, help='Batch size.')
    dataset_parser.add_argument('--version', '-V', type=int, default=0, help='The experiment version, also corresponding to the random seed to use.')
    dataset_parser.add_argument('--transfer', '-T', type=str, default="auto", help='The path of the model to load ("auto" to select the ANNc model corresponding to the current language/random seed).')
    dataset_parser.add_argument('--variant', type=str, default="classical", help='The model to use.')
    dataset_parser.add_argument('--skip', help='Skip if such a model has been trained already.', action='store_true')
    dataset_parser.add_argument('--criterion', '-C', type=str, default="relative shuffle", choices=["cosine embedding loss", "relative shuffle", "relative all", "all"], help='The training loss to use.')

    return parser, model_parser, dataset_parser

if __name__ == '__main__':
    from argparse import ArgumentParser

    # argument parsing
    parser = ArgumentParser()
    parser, model_parser, dataset_parser = add_argparse_args(parser)

    args = parser.parse_args()

    if args.transfer == "auto":
        from utils.ckpt import get_clf_in_prefix, EXCLUDED_LANGS
        print("transfer")
        if args.language not in EXCLUDED_LANGS:
            args.transfer = get_clf_in_prefix(args.dataset)[f"{args.dataset} - {args.language} - {args.version}"]

    main(args)
