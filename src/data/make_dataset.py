from datasets import load_dataset
from transformers import AutoTokenizer


def prepare_train_features(examples, tokenizer, label2id, structure_to_id):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples['question'],
        examples['context'],
        truncation="only_second",
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=True,
        is_split_into_words=True,
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["labels"] = []
    tokenized_examples["num_span"] = []
    tokenized_examples["structure"] = []
    tokenized_examples["example_id"] = []
    # tokenized_examples["word_ids"] = []
    # tokenized_examples["sequence_ids"] = []

    for i, sample_index in enumerate(sample_mapping):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        label = examples['label'][sample_index]
        word_ids = tokenized_examples.word_ids(i)
        previous_word_idx = None
        label_ids = [-100] * token_start_index

        for word_idx in word_ids[token_start_index:]:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        tokenized_examples["labels"].append(label_ids)
        tokenized_examples["num_span"].append(examples['num_span'][sample_index] / 30)
        tokenized_examples["structure"].append(structure_to_id[examples['structure'][sample_index] if 'structure' in examples else ''])
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # tokenized_examples["word_ids"].append(word_ids)
        # tokenized_examples["sequence_ids"].append(sequence_ids)
    return tokenized_examples


def create_dataset(train_file, model_str):
    data_files = {'train': train_file}
    raw_datasets = load_dataset('json', field='data', data_files=data_files)
    train_examples = raw_datasets["train"]
    column_names = raw_datasets["train"].column_names

    label_list = ["B", "I", "O"]
    label2id = {l: i for i, l in enumerate(label_list)}
    structure_list = ['Complex', 'Conjunction', 'Non-Redundant', 'Redundant', 'Share', '']
    structure_to_id = {l: i for i, l in enumerate(structure_list)}

    tokenizer = AutoTokenizer.from_pretrained(
                model_str,
                use_fast=True,
                use_auth_token=False,
                add_prefix_space=True,
            )

    train_dataset = train_examples.map(
                    lambda examples: prepare_train_features(examples, tokenizer, label2id, structure_to_id),
                    batched=True,
                    remove_columns=column_names,
                    desc="Running tokenizer on train dataset",
            )
    return train_dataset


def prepare_val_features(examples, tokenizer, label2id, structure_to_id):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples['question'],
        examples['context'],
        truncation="only_second",
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=True,
        is_split_into_words=True,
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["labels"] = []
    tokenized_examples["num_span"] = []
    tokenized_examples["structure"] = []
    tokenized_examples["example_id"] = []
    tokenized_examples["word_ids"] = []
    tokenized_examples["sequence_ids"] = []

    for i, sample_index in enumerate(sample_mapping):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        label = examples['label'][sample_index]
        word_ids = tokenized_examples.word_ids(i)
        previous_word_idx = None
        label_ids = [-100] * token_start_index

        for word_idx in word_ids[token_start_index:]:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        tokenized_examples["labels"].append(label_ids)
        tokenized_examples["num_span"].append(examples['num_span'][sample_index] / 30)
        tokenized_examples["structure"].append(structure_to_id[examples['structure'][sample_index] if 'structure' in examples else ''])
        tokenized_examples["example_id"].append(examples["id"][sample_index])
        tokenized_examples["word_ids"].append(word_ids)
        tokenized_examples["sequence_ids"].append(sequence_ids)
    return tokenized_examples


def create_val_dataset(val_file, model_str):
    data_files = {'validation': val_file}
    raw_datasets = load_dataset('json', field='data', data_files=data_files)
    val_examples = raw_datasets["validation"]
    column_names = raw_datasets["validation"].column_names

    label_list = ["B", "I", "O"]
    label2id = {l: i for i, l in enumerate(label_list)}
    structure_list = ['Complex', 'Conjunction', 'Non-Redundant', 'Redundant', 'Share', '']
    structure_to_id = {l: i for i, l in enumerate(structure_list)}

    tokenizer = AutoTokenizer.from_pretrained(
                model_str,
                use_fast=True,
                use_auth_token=False,
                add_prefix_space=True,
            )

    val_dataset = val_examples.map(
                  lambda examples: prepare_val_features(examples, tokenizer, label2id, structure_to_id),
                  batched=True,
                  remove_columns=column_names,
                  desc="Running tokenizer on val dataset",
                )
    return val_dataset