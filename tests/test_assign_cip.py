# -*- coding: utf-8 -*-


import unittest

from rdkit import Chem

import scaffound

from src.scaffound import MinMaxShortestPathOptions


class AssignCIPLAblesTestCase(unittest.TestCase):

    def setUp(self):
        self.mols = list(map(Chem.MolFromSmiles, ['P(CCCCC)(CCBr)(CCCCCC1C=CC=CC=1)(CCC1C=CC=CC=1)CCCCCCCCCCCCCC',
                                                  'S1(C(C)CCCCCCCCCCCCC1)(=O)C',
                                                  'B1(C=CC=CC1)C']))

    def test_phosphorus(self):
        self.assertEqual(scaffound.assign_cip(self.mols[0], 0), {6: 1, 28: 2, 9: 3, 1: 4, 20: 5})

    def test_sulfur(self):
        self.assertEqual(scaffound.assign_cip(self.mols[1], 0), {16: float('inf'), 15: 1, 1: 2, 17: 3})

    def test_boron(self):
        self.assertEqual(scaffound.assign_cip(self.mols[2], 0), {5: 1, 1: 2, 6: 3})


if __name__ == '__main__':
    unittest.main()
