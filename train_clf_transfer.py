
from siganalogies.config import SIG2019_LOW
from train_clf import ClfLightning as ClfLightningBase

import logging
from packaging import version

logger = logging.getLogger("")#__name__)
logger.setLevel(logging.INFO)

# configure logging at the root level of lightning
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setLevel(logging.WARNING)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add the handlers to the loggers
logger.addHandler(ch)

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from siganalogies import SIG2016_LANGUAGES, SIG2019_HIGH, SIG2019_HIGH_LOW_PAIRS, dataset_factory
from utils import prepare_data
from utils.ckpt import get_clf_in_prefix, EXCLUDED_LANGS

import os
os.environ['PYTHONHASHSEED'] = str(42)
seed_everything(42, workers=True)

MODEL_RANDOM_SEEDS = [42, 8564851, 706303, 248, 8994204, 7332146, 800, 3347863, 1402754, 7938707]

class ClfLightning(ClfLightningBase):
    def __init__(self, *args, train_clf=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_clf = train_clf

    def configure_optimizers(self):
        # @lightning method
        if self.train_clf:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        else:
            optimizer = torch.optim.Adam(self.emb.parameters(), lr=1e-3)
        return optimizer

def main(args):
    expe_name = f"clf-transfer/{args.dataset}/{args.source_language}-{args.source_dataset}-{args.language}-{args.dataset}"
    save_dir=f"logs/finetune{'-emb-only' if args.finetune_emb_only else ''}{'-tgt-enc' if args.use_tgt_enc else ''}-{args.max_epochs}" if args.finetune else "logs"

    # if the --skip is used and the experiment has already been done, abort execution
    if args.skip and os.path.exists(f"{save_dir}/{expe_name}/version_{args.version}/summary.csv"):
        logger.warning(f"{save_dir}/{expe_name}/version_{args.version}/summary.csv exists, skip")
        return

    
    use_spawn = False
    use_strategy = version.parse(pl.__version__) >= version.parse("1.5.0")

    logger.debug("Determinism (--deterministic True) does not guarantee reproducible results when changing the number of processes.")

    from pytorch_lightning.plugins import DDPSpawnPlugin, DDPPlugin, DDP2Plugin
    train_loader, val_loader, test_loader, encoder = prepare_data(
        args.dataset, args.language, args.nb_analogies_train, args.nb_analogies_val, args.nb_analogies_test,
        args.batch_size, args.force_rebuild, split_seed=MODEL_RANDOM_SEEDS[args.version], force_low=args.force_low)

    mode = "train"
    if args.source_dataset == "2019":
        mode += "-high" if args.source_language in SIG2019_HIGH else "-low"

    source_encoder = dataset_factory(dataset=args.source_dataset, language=args.source_language, mode=mode, word_encoder="char", force_rebuild=args.force_rebuild).word_encoder
    overlap = len(set(source_encoder.id_to_char).intersection(set(encoder.id_to_char)))
    src_l, tgt_l = len(source_encoder.id_to_char), len(encoder.id_to_char)
    logger.info(f"Alphabetical overlap: {overlap} of {src_l} (source, i.e., {overlap/src_l:.2%}) and {tgt_l} (target, i.e., {overlap/tgt_l:.2%}).")
    if not args.use_tgt_enc:
        train_loader.dataset.dataset.word_encoder = source_encoder
        val_loader.dataset.dataset.word_encoder = source_encoder
        test_loader.dataset.dataset.word_encoder = source_encoder
        encoder = source_encoder

    # --- Define models ---
    char_emb_size = 64

    nn = ClfLightning(char_emb_size=char_emb_size, encoder=encoder, filters=args.filters, train_clf=not args.finetune_emb_only)
    if not args.use_tgt_enc:
        try:
            state_dict = torch.load(args.ckpt, map_location="cpu")["state_dict"]
            state_dict_emb = {k[len("emb."):]: v for k, v in state_dict.items() if k.startswith("emb.")}
            nn.emb.load_state_dict(state_dict_emb)
            logger.info(f"Successfully loaded embedding from {args.ckpt}")
        except Exception: logger.warning(f"Failed loading embedding from {args.ckpt}")
    else:
        logger.info(f"Not loading embedding from {args.ckpt}, not using the same vocab.")
    try:
        state_dict = torch.load(args.ckpt, map_location="cpu")["state_dict"]
        state_dict_clf = {k[len("clf."):]: v for k, v in state_dict.items() if k.startswith("clf.")}
        nn.clf.load_state_dict(state_dict_clf)
        logger.info(f"Successfully loaded classifier network from {args.ckpt}")
    except Exception: logger.warning(f"Failed loading classifier network from {args.ckpt}")
    nn.encoder = encoder
    
    # --- Train model ---
    expe_name = f"clf-transfer/{args.dataset}/{args.source_language}-{args.language}"
    save_dir=f"logs/finetune{'-emb-only' if args.finetune_emb_only else ''}{'-tgt-enc' if args.use_tgt_enc else ''}-{args.max_epochs}" if args.finetune else "logs"
    tb_logger = pl.loggers.TensorBoardLogger(save_dir, expe_name, version=args.version)
    trainer_kwargs = {"strategy" if use_strategy else "plugins": DDPSpawnPlugin(find_unused_parameters=False) if use_spawn # causes dataloader issues
                else DDPPlugin(find_unused_parameters=False)}
    if args.finetune:
        checkpoint_callback=ModelCheckpoint(
            filename=f"clf-transfer-{args.dataset}-{args.language}-b{args.batch_size}-{{epoch:02d}}",
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

    # --- Test model ---
    with torch.no_grad():
        trainer = pl.Trainer.from_argparse_args(args,
            logger = tb_logger,
            **trainer_kwargs
        )
        nn.save_path = os.path.join(save_dir, expe_name, f"version_{args.version}")
        nn.extra_info = {
            "n_chars_source": src_l,
            "n_chars_target": tgt_l,
            "n_chars_overlap": overlap,
            "overlap_of_source": overlap/src_l,
            "overlap_of_target": overlap/tgt_l,
            "char_voc": "target" if args.use_tgt_enc else "source",
            "max_epoch": args.max_epochs,
            "best_model": args.ckpt or checkpoint_callback.best_model_path,
            "seed": MODEL_RANDOM_SEEDS[args.version],
            "seed_id": args.version,
            "lang": args.language,
            "dataset": args.dataset,
            "variant": f"finetune{'-emb-only' if args.finetune_emb_only else ''}" if args.finetune else "no finetune",
        }
        nn.common_save_path = "results/clf-transfer.pkl"
        trainer.test(nn, dataloaders=test_loader)#, ckpt_path=checkpoint_callback.best_model_path)

def add_argparse_args(parser):
    parser = pl.Trainer.add_argparse_args(parser)

    model_parser = parser.add_argument_group("Model arguments")
    model_parser.add_argument('--filters', '-f', type=int, default=128, help='The number of filters of the classification model.')

    dataset_parser = parser.add_argument_group("Dataset arguments")
    dataset_parser.add_argument('--dataset', '-d', type=str, default="2016", help='The language to train the model on.', choices=["2016", "2019"])
    dataset_parser.add_argument('--language', '-l', type=str, default="arabic", help='The language to train the model on. Use "2019" to automatically perform transfer from high to low ressource languages defined in Sigmorphon 2019.', choices=SIG2019_HIGH+SIG2019_LOW+SIG2016_LANGUAGES+["2019"])
    dataset_parser.add_argument('--force-rebuild', help='Force the re-building of the dataset file.',  action='store_true')
    dataset_parser.add_argument('--nb-analogies-test', '-t', type=int, default=50000, help='The maximum number of analogies (before augmentation) we test the model on.')
    dataset_parser.add_argument('--batch-size', '-b', type=int, default=512, help='Batch size.')
    dataset_parser.add_argument('--version', '-V', type=int, default=0, help='The experiment version, also corresponding to the random seed to use.')
    dataset_parser.add_argument('--use-tgt-enc', action='store_true')
    dataset_parser.add_argument('--skip', help='Skip if such a model has been trained already.', action='store_true')

    finetune_parser = parser.add_argument_group("Finetuning arguments")
    finetune_parser.add_argument('--finetune', help='Resume training on the target language.',  action='store_true')
    finetune_parser.add_argument('--finetune-emb-only', action='store_true')
    finetune_parser.add_argument('--force-low', action='store_true')
    finetune_parser.add_argument('--nb-analogies-train', '-n', type=int, default=50000, help='The maximum number of analogies (before augmentation) we train the model on.')
    finetune_parser.add_argument('--nb-analogies-val', '-v', type=int, default=500, help='The maximum number of analogies (before augmentation) we validate the model on.')

    source_parser = parser.add_argument_group("Source model arguments")
    source_parser.add_argument('--ckpt', type=str, help='Checkpoint.', default="")
    source_parser.add_argument('--source-language', '-sl', type=str, default="arabic", help='The language the transferred model was trained on.', choices=SIG2019_HIGH+SIG2016_LANGUAGES)
    source_parser.add_argument('--source-dataset', '-sd', type=str, default="", help='The language the transferred model was trained on. If not provided, assumed to be identical to --dataset.', choices=["2016", "2019", ""])

    return parser, model_parser, dataset_parser, finetune_parser, source_parser
    
if __name__ == '__main__':
    from argparse import ArgumentParser

    # argument parsing
    parser = ArgumentParser()
    parser, model_parser, dataset_parser, finetune_parser, source_parser = add_argparse_args(parser)
    
    args = parser.parse_args()

    assert args.ckpt
    if not args.source_dataset: args.source_dataset = args.dataset


    if args.language=="2019":
        args.dataset = "2019"
        args.source_dataset = "2019"
        for args.finetune, args.finetune_emb_only, args.max_epochs in [(False, False, 0), (True, False, 1), (True, False, 5), (True, True, 1), (True, True, 5)]:
            for args.source_language, args.language in SIG2019_HIGH_LOW_PAIRS:
                if args.source_language not in EXCLUDED_LANGS:
                    args.ckpt = get_clf_in_prefix(args.source_dataset)[f"{args.source_dataset} - {args.source_language} - {args.version}"]
                    
                    logger.warning(f"Transfering from '{args.source_dataset} - {args.source_language} - {args.version}' to '{args.language}'.")
                    logger.warning(f"Using model saved at '{args.ckpt}'.")

                    #try:rgs.source_dataset
                    main(args)
                else:
                    logger.warning(f"Ignoring '{args.source_dataset} - {args.source_language}'.")
                
    else:
        if not args.source_dataset:
            args.source_dataset = args.dataset
        if not args.ckpt:
            args.ckpt = get_clf_in_prefix(args.source_dataset)[f"{args.source_dataset} - {args.source_language} - {args.version}"]
            logger.warning(f"Using model saved at '{args.ckpt}'.")





    #try:
    main(args)