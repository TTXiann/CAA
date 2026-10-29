from eval_func.cc.bleu.bleu import Bleu
from eval_func.cc.cider.cider import Cider
from eval_func.cc.meteor.meteor import Meteor
from eval_func.cc.rouge.rouge import Rouge


def get_eval_score(references, hypotheses):
    """Corpus-level caption metrics: BLEU-1..4, METEOR, ROUGE-L, CIDEr."""
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]

    hypo = [[' '.join(hypo)] for hypo in [[str(x) for x in hypo] for hypo in hypotheses]]
    ref = [[' '.join(reft) for reft in reftmp]
           for reftmp in [[[str(x) for x in reft] for reft in reftmp] for reftmp in references]]

    score = []
    method = []
    all_scores = {}  # per-sentence scores

    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(ref, hypo)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)
        print("{} {}".format(method_i, score_i))

        if type(method_i) == list:
            all_scores['Bleu_4'] = scores_i[-1]
        else:
            all_scores[method_i] = scores_i

    return dict(zip(method, score)), all_scores


def compute_precision_recall(y_true, y_pred):
    """Per-class precision / recall for the change vs no-change classification."""
    classes = sorted(set(y_true))

    results = {}

    for cls in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp == cls)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != cls and yp == cls)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp != cls)

        results[cls] = {
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        }

    return results
