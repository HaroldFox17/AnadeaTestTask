"""
This is slightly modified version of official evaluation script for this task.
I decided to go with it since the F1 and EM metrics it calculates are already widely used for this task, so it's easy
to compare this model to existing solutions. Modifications only accommodate changes in data format and do not change the metrics
"""

import collections
import re
import string

OPTS = None


def make_qid_to_has_ans(dataset):
    qid_to_has_ans = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid_to_has_ans[qa['id']] = bool(qa['answers'])
    return qid_to_has_ans


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


def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_raw_scores(dataset, preds):
    exact_scores = {}
    f1_scores = {}
    for article in dataset:
        for qa in article['qas']:
            qid = qa['id']
            gold_answers = [a['text'] for a in qa['answers']
                            if normalize_answer(a['text'])]
            if not gold_answers:
                # For unanswerable questions, only correct answer is empty string
                gold_answers = ['']
            if qid not in [i['id'] for i in preds]:
                print('Missing prediction for %s' % qid)
                continue
            a_pred = [i['answer'] for i in preds if i['id'] == qid][0]
            # Take max over all gold answers
            exact_scores[qid] = max(compute_exact(a, p) for a in gold_answers for p in a_pred)
            f1_scores[qid] = max(compute_f1(a, p) for a in gold_answers for p in a_pred)
    return exact_scores, f1_scores


if __name__ == '__main__':
    pass
