import json
import warnings
import re
import string
import matplotlib.pyplot as plt


def get_entities(label, token):
    def _validate_chunk(chunk):
        if chunk in ['O', 'B', 'I']:
            return
        else:
            warnings.warn('{} seems not to be IOB tag.'.format(chunk))
    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []

    # check no ent
    if isinstance(label[0], list):
        for i,s in enumerate(label):
            if len(set(s)) == 1:
                chunks.append(('O', -i, -i))
    # for nested list
    if any(isinstance(s, list) for s in label):
        label = [item for sublist in label for item in sublist + ['O']]
    if any(isinstance(s, list) for s in token):
        token = [item for sublist in token for item in sublist + ['O']]

    for i, chunk in enumerate(label + ['O']):
        _validate_chunk(chunk)
        tag = chunk[0]
        if end_of_chunk(prev_tag, tag):
            chunks.append((' '.join(token[begin_offset:i]), begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag):
            begin_offset = i
        prev_tag = tag
    return chunks


def end_of_chunk(prev_tag, tag):
    chunk_end = False
    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True
    return chunk_end


def start_of_chunk(prev_tag, tag):
    chunk_start = False
    if tag == 'B':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True
    return chunk_start


def read_gold(gold_file):
    with open(gold_file, encoding='utf-8') as f:
        data = json.load(f)['data']
        golds = {}
        for piece in data:
            golds[piece['id']] = set(map(lambda x: x[0], get_entities(piece['label'],piece['context'])))
    return golds


def read_pred(pred_file):
    with open(pred_file) as f:
        preds = json.load(f)
    return preds


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


if __name__ == "__main__":
    golds = read_gold("Data/MultiSpanQA_data/valid.json")
    for k,v in golds.items():
        golds[k] = set(map(lambda x: normalize_answer(x), v))
    preds = read_pred("predict_files/pred_answers_no_I_beginning_roberta_v1.json")
    for k,v in preds.items():
        preds[k] = set(map(lambda x: normalize_answer(x), v))

    correct_answers_length = []
    missing_answers_length = []
    incorrect_answers_length = []
    for k in golds:
        gold = golds[k]
        pred = preds[k]
        for answer in gold:
            answer_length = len(answer.split())
            if answer in pred:
                correct_answers_length.append(answer_length)
            if answer not in pred:
                missing_answers_length.append(answer_length)
        for answer in pred:
            if answer not in gold:
                incorrect_answers_length.append(answer_length)

    fig, ax = plt.subplots(2, 2)
    fig.tight_layout(pad=1.0)
    ax[0, 0].hist(correct_answers_length, bins=20)
    ax[0, 0].set_title('Correct Answers Length', size=10)

    ax[0, 1].hist(missing_answers_length, bins=20)
    ax[0, 1].set_title('Missing Answers Length', size=10)

    ax[1, 0].hist(incorrect_answers_length, bins=20)
    ax[1, 0].set_title('Incorrect Answers Length', size=10)
    plt.show()
