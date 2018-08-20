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
        self.assertAlmostEqual(b.score(bleu.smooth_2), -1.64997, 4)
        self.assertAlmostEqual(b.score(bleu.smooth_3), -1.00634, 4)
        self.assertAlmostEqual(b.score(bleu.smooth_4), -2.54978, 4)


    def testInplaceAdd(self):
        candidate1 = 'the the the the the the the'
        candidate2 = 'the the cat'
        ref1 = 'the cat is on the mat'
        ref2 = 'there is a cat on the mat'

        b1 = bleuStats(candidate1, [ref1, ref2])
        b2 = bleuStats(candidate2, [ref1, ref2])

        b1 += b2

        self.assertEqual(b1.len, 10)
        self.assertEqual(b1.bmlen, 13)
        self.assertSequenceEqual(b1.match, (5.0, 1.0, 0.0, 0.0))
        self.assertSequenceEqual(b1.total, (10, 8, 6 ,4))
        self.assertAlmostEqual(b1.brevity_penalty, -0.3, 4)
