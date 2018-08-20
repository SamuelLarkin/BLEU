#!/usr/bin/env python

from __future__ import print_function

import unittest
import bleu
from bleu import bleuStats

class TestBleu(unittest.TestCase):
    def testWikipedia(self):
        candidate = 'the the the the the the the'
        ref1 = 'the cat is on the mat'
        ref2 = 'there is a cat on the mat'

        b = bleuStats(candidate, [ref1, ref2])

        #print('bleu:', b)
        #print(b.score())

        self.assertEqual(b.len, 7)
        self.assertEqual(b.bmlen, 7)
        self.assertSequenceEqual(b.match, (2.0, 0.0, 0.0, 0.0))
        self.assertSequenceEqual(b.total, (7, 6, 5, 4))
        self.assertEqual(b.brevity_penalty, 0.0)
        self.assertAlmostEqual(b.score(bleu.smooth_0), 0.0, 8)
        self.assertAlmostEqual(b.score(bleu.smooth_1), -52.1214, 4)
