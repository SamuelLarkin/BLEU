#!/usr/bin/env python

from __future__ import print_function

from itertools import izip
from itertools import starmap
from itertools import tee
from collections import Counter
from copy import deepcopy
from math import log10
from math import log
from functools import partial


def ngram(iterable, n=2):
   """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
   assert n > 0, 'Cannot create negative n-grams.'
   l = tee(iterable, n)
   for i, s in enumerate(l):
      for _ in xrange(i):
         next(s, None)
   return izip(*l)


def smooth_0(match, total, N):
    if match == 0:
        return 0.
    else:
        return log(float(match) / float(total)) / N


def smooth_1(match, total, N):
    if match == 0:
        return log(1e-30) / N
    else:
        return log(float(match) / float(total)) / N


class bleuStats:
    norder = 4
    
    def __init__(self, sentence, references):
        sentence    = sentence.strip().split()
        references  = [ s.strip().split() for s in references ]
        self._tgt_counts = [ Counter(ngram(sentence, n+1)) for n in xrange(0, self.norder) ]
        self._ref_counts = [ reduce(lambda a, b: a | Counter(ngram(b, n+1)), references, Counter()) for n in xrange(0, self.norder) ]
        assert len(self._tgt_counts) == len(self._ref_counts)
        self._tgt_counts_clipped  = [ a & b for a, b in izip(self._tgt_counts, self._ref_counts) ]
        assert len(self._tgt_counts) == len(self._tgt_counts_clipped)

        self.len     = len(sentence)
        self.bmlen   = sorted([ (abs(r_len - self.len), r_len) for r_len in map(len, references) ])[0][1]
        self.match   = [ sum(c.values(), 0.) for c in self._tgt_counts_clipped ]
        self.total   = map(lambda x: max(x, 0), xrange(self.len, self.len - self.norder, -1))


    def __isub__(self, other):
        self.len -= other.len
        self.bmlen -= other.bmlen
        self.match = [ a - b for a, b in izip(self.match, other.match) ]
        self.total = [ a - b for a, b in izip(self.total, other.total) ]
        return self


    def __iadd__(self, other):
        self.len += other.len
        self.bmlen += other.bmlen
        self.match = [ a + b for a, b in izip(self.match, other.match) ]
        self.total = [ a + b for a, b in izip(self.total, other.total) ]
        return self


    def __str__(self):
        def print_helper(array):
            return '\n'.join(('{}-ngram: {}'.format(i, str(c)) for i, c in enumerate(array, 1)))

        internal_stats = 'Counts:\n{tgt_counts}\nTotal:\n{ref_counts}\nClipped:\n{tgt_counts_clipped}\n'.format(
                tgt_counts         = print_helper(self._tgt_counts),
                ref_counts         = print_helper(self._ref_counts),
                tgt_counts_clipped = print_helper(self._tgt_counts_clipped),
                )
        stats = 'len: {len}\nbmlen: {bmlen}\nMatch: {match}\nTotal: {total}\nprecision: {precision}\nbrevity: {brevity}'.format(
                len       = self.len,
                bmlen     = self.bmlen,
                match     = self.match,
                total     = self.total,
                precision = [ float(m) / t if t > 0 else 0. for m, t in zip(self.match, self.total) ],
                brevity   = self.brevity_penalty,
                )
        return internal_stats + stats
        

    @property
    def brevity_penalty(self):
        return min(1. - float(self.bmlen) / float(self.len), 0.)


    def score(self, smoothing=smooth_1):
        """
        Computes the log BLEU score for these stats; that is:
        \log BLEU = min(1 - bmlength / length, 0) + \sum_{n=1}^{N} (1/N) \log(match_n / total_n)
        The generalized BLEU score is actually given by
        \log BLEU = min(1 - bmlength / length, 0) + \sum_{n=1}^{N} w_n \log(match_n / total_n)
        for some choice of weights (w_n).  Generally, the weights are 1/N, which is
        what is used here.

        """
        return sum(starmap(partial(smoothing, N=self.norder), izip(self.match, self.total)),
                self.brevity_penalty)
#        return sum(map(lambda (m, t): smooting(m, t, self.norder), izip(self.match, self.total)),
#                self.brevity_penalty)





if __name__ == '__main__':
    candidate = 'the the the the the the the'
    ref1 = 'the cat is on the mat'
    ref2 = 'there is a cat on the mat'

    bleu = bleuStats(candidate, [ref1, ref2])

    print('bleu:', bleu)
    print(bleu.score())

    bleu1 = bleuStats('the the cat', [ref1, ref2])
    print('bleu:', bleu1)
    print(bleu1.score())

    bleu1 += bleu
    print('bleu:', bleu1)
    print(bleu1.score())
