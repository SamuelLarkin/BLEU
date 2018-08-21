#!/usr/bin/env python

from __future__ import print_function

from collections import Counter
from copy import deepcopy
from functools import partial
from itertools import izip
from itertools import starmap
from itertools import tee
from math import ceil
from math import exp
from math import log
from random import choice

from tqdm import trange


epsilon = 1e-30


def ngram(iterable, n=2):
   """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
   assert n > 0, 'Cannot create negative n-grams.'
   l = tee(iterable, n)
   for i, s in enumerate(l):
      for _ in xrange(i):
         next(s, None)
   return izip(*l)



def smooth_0(bleustats):
    """
    no smoothing
    """
    N = bleustats.norder
    result = bleustats.brevity_penalty
    for match, total in bleustats.stats():
        if match == 0:
            return 0.0
        else:
            assert match >= 0
            assert total > 0
            result += log(float(match) / float(total)) / N
    return result



def smooth_1(bleustats):
    """
    Replace 0 by epsilon
    """
    def helper(match, total, N):
        if match == 0:
            return log(epsilon) / N
        else:
            assert match >= 0
            assert total > 0
            return log(float(match) / float(total)) / N

    return sum(starmap(partial(helper, N=bleustats.norder), bleustats.stats()), bleustats.brevity_penalty)



def smooth_2(bleustats):
    """
    Increase the count by 1 for all n-grams with n>1 (cf. Lin & Och, Coling 2004)
    """
    assert bleustats.total[0] > 0
    N = bleustats.norder
    result = bleustats.brevity_penalty
    if bleustats.match[0] > 0:
        # 1-gram: count is not changed
        # higher n-grams: all counts and the total are increased by 1
        for (match, total), smoothing in izip(bleustats.stats(), [0.0] + [1.0] * (N - 1)):
            result += log((float(match) + smoothing) / (float(total) + smoothing)) / N
    else:
        result += log(epsilon) / N

    return result



def smooth_3(bleustats):
    """
    score = sum_{i=1}N {i-BLEU(x,y) / 2^{N-i+1}}
    """
    def helper(match, total, N):
        if match == 0:
            return log(epsilon) / N
        else:
            return log(float(match) / float(total)) / N

    N = bleustats.norder
    stats = list(bleustats.stats())
    result = 0.0
    for i in xrange(1, N+1):
        sub_result = sum(starmap(partial(helper, N=N), stats[:i]), bleustats.brevity_penalty)
        result += result + (exp(sub_result) / pow(2.0, (N-i+1.0)) );

    return log(result)



def smooth_4(bleustats):
    """
    New smoothing from mteval-v13a.pl.
    Documentation from mteval-v13a.pl:
    The smoothing is computed by taking 1 / ( 2^k ), instead of 0, for each precision score whose matching n-gram count is null
    k is 1 for the first 'n' value for which the n-gram match count is null
    For example, if the text contains:
      - one 2-gram match
      - and (consequently) two 1-gram matches
    the n-gram count for each individual precision score would be:
      - n=1  =>  prec_count = 2     (two unigrams)
      - n=2  =>  prec_count = 1     (one bigram)
      - n=3  =>  prec_count = 1/2   (no trigram,  taking 'smoothed' value of 1 / ( 2^k ), with k=1)
      - n=4  =>  prec_count = 1/4   (no fourgram, taking 'smoothed' value of 1 / ( 2^k ), with k=2)
    """
    class helper:
        def __init__(self, N):
            self.smooth = 1.0
            self.N = N

        def __call__(self, match, total):
            if total > 0:
                assert match >= 0
                if match == 0:
                    self.smooth *= 2
                    X = 1.0 / (self.smooth * float(total))
                else:
                    X = float(match) / float(total)
                return log(X) / self.N
            else:
                return 0.0

    # TODO: Don't we need a brevity penalty?
    return sum(starmap(helper(N=bleustats.norder), bleustats.stats()), 0.0)



class bleuStats:
    norder = 4

    def __init__(self, sentence='', references=['']):
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


    def __add__(self, other):
        result = deepcopy(self)
        result += other
        return result


    def _format_stats(self):
        def format_ngram(n, match, total):
            return '{n}-gram (match/total) {match}/{total} {precision:0.6f}'.format(
                    n=n,
                    match=match,
                    total=total,
                    precision=float(match)/float(total) if total > 0 else 0.)

        def format_ngrams(stats):
            return '\n'.join(format_ngram(n, match, total) for n, (match, total) in enumerate(stats, 1))

        stat_format  = '{ngrams}\n'
        stat_format += 'Sentence length: {len}; Best-match sentence length: {bmlen}\n'
        stat_format += 'Brevity penalty: exp({brevity:0.6f}) = {exp_brevity:0.6f}'

        return stat_format.format(
                ngrams      = format_ngrams(self.stats()),
                len         = self.len,
                bmlen       = self.bmlen,
                brevity     = self.brevity_penalty,
                exp_brevity = exp(self.brevity_penalty),
                )


    def _format_internal_stats(self):
        def print_helper(array):
            return '\n'.join(('{}-ngram: {}'.format(i, str(c)) for i, c in enumerate(array, 1)))

        stat_format  = 'Counts:\n{tgt_counts}\n'
        stat_format += 'Total:\n{ref_counts}\n'
        stat_format += 'Clipped:\n{tgt_counts_clipped}\n'

        return stat_format.format(
                tgt_counts         = print_helper(self._tgt_counts),
                ref_counts         = print_helper(self._ref_counts),
                tgt_counts_clipped = print_helper(self._tgt_counts_clipped),
                )


    def __str__(self):
        return self._format_stats()
        #return self._format_internal_stats() + self._format_stats()


    @property
    def brevity_penalty(self):
        return min(1. - float(self.bmlen) / float(self.len), 0.)


    def stats(self):
        """
        Returns an iterator of (match, total).
        """
        return izip(self.match, self.total)


    def score(self, smoothing=smooth_1):
        """
        Computes the log BLEU score for these stats; that is:
        \log BLEU = min(1 - bmlength / length, 0) + \sum_{n=1}^{N} (1/N) \log(match_n / total_n)
        The generalized BLEU score is actually given by
        \log BLEU = min(1 - bmlength / length, 0) + \sum_{n=1}^{N} w_n \log(match_n / total_n)
        for some choice of weights (w_n).  Generally, the weights are 1/N, which is
        what is used here.

        """
        return smoothing(self)


    def bleu(self, smoothing=smooth_1):
        return exp(self.score(smoothing))



def bootstrapConfInterval(bleus, conf=0.95, m=1000):
    bleu = sum(bleus, bleuStats()).bleu()
    deltas = []
    for _ in trange(m, desc='Computing the confidence'):
        t = [ choice(bleus) for n in bleus ]
        deltas.append(abs(bleu - sum(t, bleuStats()).bleu()))
    deltas.sort()
    return deltas[int(ceil(m * conf) - 1)]





def get_args():
    """Command line argument processing."""
    from argparse import ArgumentParser

    usage="bleu.py [options] translation ref1 [ref2 ...]"
    help="""
    Computes the BLEU score for the set of translations in testfile, using the
    reference files ref1, ... , refn. Each file should have one sentence per line,
    and the sentences in testfile should match line for line with the sentences in
    each reference file.abs
    """

    parser = ArgumentParser(usage=usage, description=help)

    # TODO: we need to describe what are smooth{0,1,2,3,4}
    # This could be helpful https://stackoverflow.com/a/49999185
    parser.add_argument('-s',
            dest='smoothing',
            metavar='SMooThing',
            nargs="?",
            choices=(smooth_0.__name__, smooth_1.__name__, smooth_2.__name__, smooth_3.__name__, smooth_4.__name__, ),
            const=smooth_1,
            default=smooth_1,
            help="one of %(choices) -s alone implies %(const)s [%(default)s]")

    parser.add_argument("translation_file",
            type=open,
            help="translation file")
    parser.add_argument("reference_files",
            nargs='+',
            type=open,
            help="reference files")

    cmd_args = parser.parse_args()

    # info, verbose, debug all print to stderr.
    #print("arguments are:")
    #for arg in cmd_args.__dict__:
    #   print("  {0} = {1}".format(arg, getattr(cmd_args, arg)))
    return cmd_args



def main():
    bleu = bleuStats()

    args = get_args()

    bleus = [ bleuStats(translation, references) for translation, references in izip(args.translation_file, izip(*args.reference_files)) ]
    bleu  = sum(bleus, bleuStats())
    confidence = bootstrapConfInterval(bleus)

    print(bleu)
    print('Score: {score:0.6f}'.format(score = bleu.score(args.smoothing)))
    print('BLEU score: {bleu:0.6f}'.format (bleu = bleu.bleu(args.smoothing)))
    print('Human readable BLEU: {readable:2.2f}'.format(readable = 100 * bleu.bleu(args.smoothing)))
    print(confidence)




if __name__ == '__main__':
    main()
