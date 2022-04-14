from siganalogies import dataset_factory
from siganalogies.config import SIG2016_LANGUAGES
from siganalogies.utils import enrich, generate_negative
import os
import random

for lang in SIG2016_LANGUAGES:
    dataset = dataset_factory(language=lang, mode="train", word_encoding=None)
    indices = random.choices(range(len(dataset)), k=1000)

    valid, invalid = [], []
    for i in indices:
        a,b,c,d = dataset[i]
        for a,b,c,d in enrich(a,b,c,d):
            valid.append('\t'.join([a,b,c,d]))
        for a,b,c,d in generate_negative(a,b,c,d):
            invalid.append('\t'.join([a,b,c,d]))

    if not os.path.exists("test_examples"):
        os.mkdir("test_examples")
    if not os.path.exists(f"test_examples/{lang}"):
        os.mkdir(f"test_examples/{lang}")
    with open(f"test_examples/{lang}/valid.txt", "w") as f:
        f.write('\n'.join(valid))
    with open(f"test_examples/{lang}/invalid.txt", "w") as f:
        f.write('\n'.join(invalid))

    print(f"Wrote {len(valid)} samples for {lang}.")