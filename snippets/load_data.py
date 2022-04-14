
from torch.utils.data import DataLoader, Subset





if __name__ == "__main__":
    train_dataset = Task1Dataset(language=language, mode="train", word_encoding="char") 
    if language == "japanese":
        japanese_train_analogies, japanese_test_analogies = train_test_split(train_dataset.analogies, test_size=0.3, random_state = 42)

        test_dataset = copy(train_dataset)
        test_dataset.analogies = japanese_test_analogies

        train_dataset.analogies = japanese_train_analogies
    else:
        test_dataset = Task1Dataset(language=language, mode="test", word_encoding="char")

    # Generate special characters and unify the dictionaries of the training and test sets
    voc = train_dataset.char_voc_id
    BOS_ID = len(voc) # (max value + 1) is used for the beginning of sequence value
    EOS_ID = len(voc) + 1 # (max value + 2) is used for the end of sequence value

    test_dataset.char_voc = train_dataset.char_voc
    test_dataset.char_voc_id = voc

    #subsets
    if len(test_dataset) > nb_analogies:
        test_indices = list(range(len(test_dataset)))
        test_sub_indices = rd.sample(test_indices, nb_analogies)
        test_subset = Subset(test_dataset, test_sub_indices)
    else:
        test_subset = test_dataset

    
    #load data
    test_dataloader = DataLoader(test_subset, shuffle = True, collate_fn = partial(collate, bos_id = BOS_ID, eos_id = EOS_ID))