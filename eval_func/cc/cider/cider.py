# CIDEr metric (Vedantam, Zitnick, Parikh): http://arxiv.org/abs/1411.5726

from eval_func.cc.cider.cider_scorer import CiderScorer

class Cider:
    """Compute the CIDEr metric."""
    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma

    def compute_score(self, gts, res):
        """Return corpus CIDEr score and per-image scores from tokenised hypotheses and references."""

        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)

        for i in range(len(res)):
            hypo = res[i]
            ref = gts[i]

            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

            cider_scorer += (hypo[0], ref)

        (score, scores) = cider_scorer.compute_score()

        return score, scores

    def method(self):
        return "CIDEr"