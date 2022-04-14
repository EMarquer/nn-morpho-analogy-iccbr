from config import VOCAB_PATH
from siganalogies import dataset_factory
from siganalogies.config import SIG2016_LANGUAGES
from siganalogies.utils import enrich

for lang in SIG2016_LANGUAGES:
    dataset = dataset_factory(language=lang, mode="train", word_encoding=None)
    voc = dataset.char_voc

    if lang != "japanese":
        dataset = dataset_factory(language=lang, mode="test", word_encoding=None)

    voc = list(voc)
    voc.sort()

    with open(VOCAB_PATH.format(language = lang), "w") as f:
        f.write('\n'.join(voc))

    print(f"Wrote {len(voc)} words for the vocabulary of {lang}.")