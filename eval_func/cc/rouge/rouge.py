#!/usr/bin/env python

# ROUGE-L metric as described by Lin and Hovey (2004).


import numpy as np


def my_lcs(string, sub):
    """Return the length of the longest common subsequence of two token lists."""

    if (len(string) < len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):

        for i in range(1, len(string) + 1):

            if (string[i - 1] == sub[j - 1]):

                lengths[i][j] = lengths[i - 1][j - 1] + 1

            else:

                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(string)][len(sub)]


class Rouge():
    '''ROUGE-L scorer for candidate sentences.'''

    def __init__(self):

        # vrama91: updated the value below based on discussion with Hovey

        self.beta = 1.2

    def calc_score(self, candidate, refs):

        """Return the ROUGE-L score of one candidate against its references."""

        assert (len(candidate) == 1)

        assert (len(refs) > 0)

        prec = []

        rec = []

        token_c = candidate[0].split(" ")

        for reference in refs:
            token_r = reference.split(" ")

            lcs = my_lcs(token_r, token_c)

            prec.append(lcs / float(len(token_c)))

            rec.append(lcs / float(len(token_r)))

        prec_max = max(prec)

        rec_max = max(rec)

        if (prec_max != 0 and rec_max != 0):

            score = ((1 + self.beta ** 2) * prec_max * rec_max) / float(rec_max + self.beta ** 2 * prec_max)

        else:

            score = 0.0

        return score

    def compute_score(self, references, hypotheses):

        """Return mean ROUGE-L score over all candidates plus the per-candidate scores."""

        score = []

        for i in range(len(hypotheses)):
            hypo = hypotheses[i]
            ref = references[i]

            score.append(self.calc_score(hypo, ref))

            assert (type(hypo) is list)

            assert (len(hypo) == 1)

            assert (type(ref) is list)

            assert (len(ref) > 0)

        average_score = np.mean(np.array(score))

        return average_score, score

    def method(self):

        return "Rouge"