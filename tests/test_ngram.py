#!/usr/bin/env python

from __future__ import print_function

import unittest
from bleu import ngram

class TestNgram(unittest.TestCase):
    def test_invalid_n(self):
        with self.assertRaisesRegexp(Exception, 'Cannot create negative n-grams.') as context:
            ngram('a b c d e'.split(), 0)


    def test_1gram(self):
        grams = list(ngram(''.split(), 1))
        self.assertSequenceEqual(grams, [])

        grams = list(ngram('a'.split(), 1))
        self.assertSequenceEqual(grams, [('a',)])

        grams = list(ngram('a b c d e'.split(), 1))
        self.assertSequenceEqual(grams, [('a',), ('b',), ('c',), ('d',), ('e',)])


    def test_2gram(self):
        grams = list(ngram(''.split(), 2))
        self.assertSequenceEqual(grams, [])

        grams = list(ngram('a'.split(), 2))
        self.assertSequenceEqual(grams, [])

        grams = list(ngram('a b'.split(), 2))
        self.assertSequenceEqual(grams, [('a', 'b')])

        grams = list(ngram('a b c d e'.split(), 2))
        self.assertSequenceEqual(grams, [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e')])


    def test_3gram(self):
        grams = list(ngram(''.split(), 3))
        self.assertSequenceEqual(grams, [])

        grams = list(ngram('a b'.split(), 3))
        self.assertSequenceEqual(grams, [])

        grams = list(ngram('a b c'.split(), 3))
        self.assertSequenceEqual(grams, [('a', 'b', 'c')])

        grams = list(ngram('a b c d e'.split(), 3))
        self.assertSequenceEqual(grams, [('a', 'b', 'c'), ('b', 'c', 'd'), ('c', 'd', 'e')])


    def test_4gram(self):
        grams = list(ngram(''.split(), 4))
        self.assertSequenceEqual(grams, [])

        grams = list(ngram('a b c'.split(), 4))
        self.assertSequenceEqual(grams, [])

        grams = list(ngram('a b c d'.split(), 4))
        self.assertSequenceEqual(grams, [('a', 'b', 'c', 'd')])

        grams = list(ngram('a b c d e'.split(), 4))
        self.assertSequenceEqual(grams, [('a', 'b', 'c', 'd'), ('b', 'c', 'd', 'e')])




if __name__ == '__main__':
    unittest.main()
