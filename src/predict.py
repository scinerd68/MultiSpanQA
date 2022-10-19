import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import json
from tqdm import tqdm
from data.make_dataset import create_val_dataset
from models.model import MultiSpanQATagger


def find_spans(pred_labels, item):
    # Find the start of the context
    token_start_index = 0
    sequence_ids = item['sequence_ids'][0]
    while sequence_ids[token_start_index] != 1:
        token_start_index += 1
    word_ids = item['word_ids'][0]
    previous_word_id = None
    answer_word_id = []
    spans = []
    for idx, pred in enumerate(pred_labels[1:], start=1):
        current_word_id = word_ids[idx]
        if current_word_id is None:
            previous_word_id = current_word_id
            continue
        # Ignore pad token
        if item['attention_mask'][0][idx] == 0:
            previous_word_id = current_word_id
            continue
        # if current token belongs to a new word
        if current_word_id != previous_word_id:
            # Get B-tag of the new word
            if pred == 0:
                spans.append([idx])
                answer_word_id.append(current_word_id)
            # Skip O-tag
            elif pred == 2:
                previous_word_id = current_word_id
                continue
            elif pred == 1:
                if len(spans) == 0:
                    # Uncomment this to include tag I without O at beginning
                    # spans.append([idx])
                    # answer_word_id.append(current_word_id)
                    # Uncomment this to exclude tag I without O at beginning
                    previous_word_id = current_word_id
                    continue
                # if current word is the next word of the last word in the last answer
                elif current_word_id == answer_word_id[-1] + 1:
                    spans[-1].append(idx)
                else:
                    # Uncomment this to include tag I without O at beginning
                    # spans.append([idx])
                    # answer_word_id.append(current_word_id)
                    # Uncomment this to exclude tag I without O at beginning
                    previous_word_id = current_word_id
                    continue
                answer_word_id.append(current_word_id)

        # if current token is continuation of a word
        if current_word_id == previous_word_id:
            # skip if there are no answer yet
            if len(spans) == 0:
                previous_word_id = current_word_id
                continue
            # if current word is not the last word in the last answer then skip
            if current_word_id != answer_word_id[-1]:
                previous_word_id = current_word_id
                continue
            # Add continued token of the word in the answer
            else:
                spans[-1].append(idx)
                answer_word_id.append(current_word_id)
        previous_word_id = current_word_id
    return spans


def decode_answer(spans, item, tokenizer):
    answers = []
    input_ids = item['input_ids'][0]
    for span in spans:
        start = span[0]
        end = span[-1] 
        answer = [token_id.item() for token_id in input_ids[start:end+1]]
        answer = tokenizer.decode(answer)
        answers.append(answer)
    return answers


def convert_to_tensor(batch):
    new_batch = {}
    new_batch['example_id'] = [example['example_id'] for example in batch]
    new_batch['num_span'] = torch.Tensor([example['num_span'] for example in batch])
    new_batch['structure'] = torch.LongTensor([example['structure'] for example in batch])
    new_batch['input_ids'] = torch.stack([torch.LongTensor(example['input_ids']) for example in batch])
    # new_batch['token_type_ids'] = torch.stack([torch.LongTensor(example['token_type_ids']) for example in batch])
    new_batch['attention_mask'] = torch.stack([torch.LongTensor(example['attention_mask']) for example in batch])
    new_batch['labels'] = torch.stack([torch.LongTensor(example['labels']) for example in batch])
    new_batch['word_ids'] = [example['word_ids'] for example in batch]
    new_batch['sequence_ids'] = [example['sequence_ids'] for example in batch]
    return new_batch


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load model
    model = MultiSpanQATagger()
    model = model.to(device)
    model = torch.load('models/multi_span_roberta_qa.pth',
                   map_location = torch.device(device))
    # Create val dataset
    val_dataset = create_val_dataset("D:/ComputerScience/BachKhoa/NLPLab/MultiSpanQA/data/MultiSpanQA_data/valid.json",
                                       "roberta-base")
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=convert_to_tensor)
    # Eval
    tokenizer = AutoTokenizer.from_pretrained(
                'roberta-base',
                use_fast=True,
                use_auth_token=False,
                add_prefix_space=True,
            )
    all_answers = {}
    for example in tqdm(val_loader):
        model.eval()
        out = model(example['input_ids'].to(device),
                    example['attention_mask'].to(device))
        pred_labels = torch.argmax(out['logits'], dim=2)[0]
        spans = find_spans(pred_labels, example)
        answers = decode_answer(spans, example, tokenizer)
        example_id = example['example_id'][0]
        if example_id not in all_answers:
            all_answers[example_id] = answers
        else:
            all_answers[example_id].extend(answers)

    with open('results/pred_answers_no_I_beginning_roberta_v2.json', 'w', encoding='utf-8') as f:
        json.dump(all_answers, f, ensure_ascii=False, indent=4)