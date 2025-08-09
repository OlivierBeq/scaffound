import unittest
from itertools import chain

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.rdBase import BlockLogs

import scaffound


def is_resonance_form(mol1: Chem.Mol, mol2: Chem.Mol) -> bool:
    if Chem.MolToSmiles(mol1, canonical=True) == Chem.MolToSmiles(mol2, canonical=True):
        return True
    for res in Chem.ResonanceMolSupplier(mol1):
        if Chem.MolToSmiles(res, canonical=True) == Chem.MolToSmiles(mol2, canonical=True):
            return True
    return False


class OriginalMolecularAnatomyTestCase(unittest.TestCase):

    def setUp(self):
        from .constants import COX2_SCAFFOLDS, NEW_SCAFFFOLDS
        from rdkit import Chem
        with BlockLogs():
            self.cox2 = COX2_SCAFFOLDS.assign(mol=COX2_SCAFFOLDS.molecule_smiles.apply(Chem.MolFromSmiles))
            self.others = NEW_SCAFFFOLDS.assign(mol=NEW_SCAFFFOLDS.molecule_smiles.apply(Chem.MolFromSmiles))
        self.opts = scaffound.paths.MinMaxShortestPathOptions(original_algorithm=True)

    def test_cox2_basic_scaffolds(self):
        # for i, row in self.cox2.iterrows():
        #     print(i,
        #           row.molecule_smiles,
        #           Chem.MolToSmiles(Chem.MolFromSmiles(row.basic_scaffold_smiles), canonical=True),
        #           Chem.MolToSmiles(scaffound.get_basic_scaffold(row.mol), canonical=True))
        #     self.assertTrue(Chem.MolToSmiles(Chem.MolFromSmiles(row.basic_scaffold_smiles), canonical=True) == Chem.MolToSmiles(scaffound.get_basic_scaffold(row.mol), canonical=True))
        basic_scaffolds = self.cox2.mol.apply(scaffound.get_basic_scaffold).apply(Chem.MolToSmiles, canonical=True)
        np.testing.assert_array_equal(self.cox2.basic_scaffold_smiles.apply(Chem.MolFromSmiles).apply(Chem.MolToSmiles, canonical=True).values,
                                      basic_scaffolds.values)

    def test_cox2_decorated_scaffolds(self):
        # for i, row in self.cox2.iterrows():
        #     print(i,
        #           row.molecule_smiles, end=' ', flush=True)
        #     print(Chem.MolToSmiles(Chem.MolFromSmiles(row.decorated_scaffold_smiles), canonical=True),
        #           Chem.MolToSmiles(scaffound.get_decorated_scaffold(row.mol), canonical=True))
        #     self.assertTrue(Chem.MolToSmiles(Chem.MolFromSmiles(row.decorated_scaffold_smiles), canonical=True) == Chem.MolToSmiles(scaffound.get_decorated_scaffold(row.mol), canonical=True))
        decorated_scaffolds = self.cox2.mol.apply(scaffound.get_decorated_scaffold).apply(Chem.MolToSmiles, canonical=True)
        np.testing.assert_array_equal(self.cox2.decorated_scaffold_smiles.apply(Chem.MolFromSmiles).apply(Chem.MolToSmiles, canonical=True).values,
                                      decorated_scaffolds.values)

    def test_cox2_augmented_scaffolds(self):
        # passed = 0
        # for i, row in self.cox2.iterrows():
        #     print('________\n', 'newline', i,
        #           row.molecule_smiles, flush=True)
        #     expected = Chem.MolToSmiles(Chem.MolFromSmiles(row.augmented_scaffold_smiles), canonical=True)
        #     obtained = Chem.MolToSmiles(scaffound.get_augmented_scaffold(row.mol, opts=self.opts), canonical=True)
        #     print('expected:', expected,
        #           'obtained:', obtained)
        #     if expected == obtained:
        #         passed += 1
        #     self.assertTrue(expected == obtained)
        # print(f'PASSED: {passed} /{len(self.cox2)}')
        augmented_scaffolds = self.cox2.mol.apply(scaffound.get_augmented_scaffold, opts=self.opts).apply(Chem.MolToSmiles, canonical=True)
        np.testing.assert_array_equal(self.cox2.augmented_scaffold_smiles.apply(Chem.MolFromSmiles).apply(Chem.MolToSmiles, canonical=True).values,
                                      augmented_scaffolds.values)

    def test_cox2_basic_framework(self):
        # for i, row in self.cox2.iterrows():
        #     print(i,
        #           row.molecule_smiles,
        #           Chem.MolToSmiles(Chem.MolFromSmiles(row.basic_framework_smiles), canonical=True),
        #           Chem.MolToSmiles(scaffound.get_basic_framework(row.mol), canonical=True))
        #     self.assertTrue(Chem.MolToSmiles(Chem.MolFromSmiles(row.basic_framework_smiles), canonical=True) == Chem.MolToSmiles(scaffound.get_basic_framework(row.mol), canonical=True))
        basic_frameworks = self.cox2.mol.apply(scaffound.get_basic_framework).apply(Chem.MolToSmiles, canonical=True)
        np.testing.assert_array_equal(self.cox2.basic_framework_smiles.apply(Chem.MolFromSmiles).apply(Chem.MolToSmiles).values,
                                      basic_frameworks.values)

    def test_cox2_basic_wireframe(self):
        # for i, row in self.cox2.iterrows():
        #     print(i,
        #           row.molecule_smiles,
        #           Chem.MolToSmiles(Chem.MolFromSmiles(row.basic_wireframe_smiles), canonical=True),
        #           Chem.MolToSmiles(scaffound.get_basic_wireframe(row.mol), canonical=True))
        #     self.assertTrue(Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(row.basic_wireframe_smiles), canonical=True)), canonical=True) == Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(scaffound.get_basic_wireframe(row.mol), canonical=True)), canonical=True))
        basic_wireframe = self.cox2.mol.apply(scaffound.get_basic_wireframe).apply(Chem.MolToSmiles, canonical=True).apply(Chem.MolFromSmiles).apply(Chem.MolToSmiles, canonical=True)
        np.testing.assert_array_equal(self.cox2.basic_wireframe_smiles.apply(Chem.MolFromSmiles).apply(Chem.MolToSmiles, canonical=True).values,
                                      basic_wireframe.values)

    def test_cox2_decorated_framework(self):
        # for i, row in self.cox2.iterrows():
        #     print(i, end=' ')
        #     print(row.molecule_smiles, end=' ')
        #     print(Chem.MolToSmiles(Chem.MolFromSmiles(row.decorated_framework_smiles), canonical=True), end=' ')
        #     print(Chem.MolToSmiles(scaffound.get_decorated_framework(row.mol), canonical=True))
        #     self.assertTrue(Chem.MolToSmiles(Chem.MolFromSmiles(row.decorated_framework_smiles), canonical=True) == Chem.MolToSmiles(scaffound.get_decorated_framework(row.mol), canonical=True))
        decorated_frameworks = self.cox2.mol.apply(scaffound.get_decorated_framework).apply(Chem.MolToSmiles, canonical=True)
        np.testing.assert_array_equal(self.cox2.decorated_framework_smiles.apply(Chem.MolFromSmiles).apply(Chem.MolToSmiles, canonical=True).values,
                                      decorated_frameworks.values)

    def test_cox2_decorated_wireframe(self):
        # for i, row in self.cox2.iterrows():
        #     print(i,
        #           row.molecule_smiles,
        #           Chem.MolToSmiles(Chem.MolFromSmiles(row.decorated_wireframe_smiles), canonical=True),
        #           Chem.MolToSmiles(scaffound.get_decorated_wireframe(row.mol)))
        #     self.assertTrue(Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(row.decorated_wireframe_smiles), canonical=True)), canonical=True) == Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(scaffound.get_generic_graph(scaffound.get_saturated_graph(scaffound.get_decorated_scaffold(row.mol))), canonical=True)), canonical=True))
        decorated_wireframe = self.cox2.mol.apply(scaffound.get_decorated_wireframe).apply(Chem.MolToSmiles, canonical=True).apply(Chem.MolFromSmiles).apply(Chem.MolToSmiles, canonical=True)
        np.testing.assert_array_equal(self.cox2.decorated_wireframe_smiles.apply(Chem.MolFromSmiles).apply(Chem.MolToSmiles, canonical=True).values,
                                      decorated_wireframe.values)

    def test_cox2_augmented_framework(self):
        # for i, row in self.cox2.iterrows():
        #     print(i,
        #           row.molecule_smiles,
        #           Chem.MolToSmiles(Chem.MolFromSmiles(row.augmented_framework_smiles), canonical=True),
        #           Chem.MolToSmiles(scaffound.get_augmented_framework(row.mol, opts=self.opts), canonical=True))
        #     self.assertTrue(Chem.MolToSmiles(Chem.MolFromSmiles(row.augmented_framework_smiles), canonical=True) == Chem.MolToSmiles(scaffound.get_augmented_framework(row.mol, opts=self.opts), canonical=True))
        augmented_frameworks = self.cox2.mol.apply(scaffound.get_augmented_framework, opts=self.opts).apply(Chem.MolToSmiles, canonical=True)
        np.testing.assert_array_equal(self.cox2.augmented_framework_smiles.apply(Chem.MolFromSmiles).apply(Chem.MolToSmiles, canonical=True).values,
                                      augmented_frameworks.values)

    def test_cox2_augmented_wireframe(self):
        # for i, row in self.cox2.iterrows():
        #     print(i,
        #           row.molecule_smiles,
        #           Chem.MolToSmiles(Chem.MolFromSmiles(row.augmented_wireframe_smiles), canonical=True),
        #           Chem.MolToSmiles(scaffound.get_augmented_wireframe(row.mol, opts=self.opts)))
        #     self.assertTrue(Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(row.augmented_wireframe_smiles), canonical=True)), canonical=True) == Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(scaffound.get_augmented_wireframe(row.mol, opts=self.opts), canonical=True)), canonical=True))
        augmented_wireframe = self.cox2.mol.apply(scaffound.get_augmented_wireframe, opts=self.opts).apply(Chem.MolToSmiles, canonical=True).apply(Chem.MolFromSmiles).apply(Chem.MolToSmiles, canonical=True)
        np.testing.assert_array_equal(self.cox2.augmented_wireframe_smiles.apply(Chem.MolFromSmiles).apply(Chem.MolToSmiles, canonical=True).values,
                                      augmented_wireframe.values)

    def test_additional_basic_scaffolds(self):
        # for i, row in self.others.iterrows():
        #     print(i,
        #           row.molecule_smiles,
        #           Chem.MolToSmiles(Chem.MolFromSmiles(row.basic_scaffold_smiles), canonical=True),
        #           Chem.MolToSmiles(scaffound.get_basic_scaffold(row.mol), canonical=True))
        #     self.assertTrue(Chem.MolToSmiles(Chem.MolFromSmiles(row.basic_scaffold_smiles), canonical=True) == Chem.MolToSmiles(scaffound.get_basic_scaffold(row.mol), canonical=True))
        basic_scaffolds = self.others.mol.apply(scaffound.get_basic_scaffold).apply(Chem.MolToSmiles, canonical=True)
        np.testing.assert_array_equal(self.others.basic_scaffold_smiles.apply(Chem.MolFromSmiles).apply(Chem.MolToSmiles, canonical=True).values,
                                      basic_scaffolds.values)

    def test_additional_decorated_scaffolds(self):
        # for i, row in self.others.iterrows():
        #     print(i,
        #           row.molecule_smiles, end=' ', flush=True)
        #     print(Chem.MolToSmiles(Chem.MolFromSmiles(row.decorated_scaffold_smiles), canonical=True),
        #           Chem.MolToSmiles(scaffound.get_decorated_scaffold(row.mol), canonical=True))
        #     self.assertTrue(Chem.MolToSmiles(Chem.MolFromSmiles(row.decorated_scaffold_smiles), canonical=True) == Chem.MolToSmiles(scaffound.get_decorated_scaffold(row.mol), canonical=True))
        decorated_scaffolds = self.others.mol.apply(scaffound.get_decorated_scaffold).apply(Chem.MolToSmiles, canonical=True)
        np.testing.assert_array_equal(self.others.decorated_scaffold_smiles.apply(Chem.MolFromSmiles).apply(Chem.MolToSmiles, canonical=True).values,
                                      decorated_scaffolds.values)

    def test_additional_augmented_scaffolds(self):
        # for i, row in self.others.iterrows():
        #     print('________\n', 'newline', i,
        #           row.molecule_smiles, flush=True)
        #     print('expected:', Chem.MolToSmiles(Chem.MolFromSmiles(row.augmented_scaffold_smiles), canonical=True),
        #           'obtained:', Chem.MolToSmiles(scaffound.get_augmented_scaffold(row.mol, opts=self.opts), canonical=True))
        #     self.assertTrue(Chem.MolToSmiles(Chem.MolFromSmiles(row.augmented_scaffold_smiles), canonical=True) == Chem.MolToSmiles(scaffound.get_augmented_scaffold(row.mol, opts=self.opts), canonical=True))
        augmented_scaffolds = self.others.mol.apply(scaffound.get_augmented_scaffold, opts=self.opts).apply(Chem.MolToSmiles, canonical=True)
        np.testing.assert_array_equal(self.others.augmented_scaffold_smiles.apply(Chem.MolFromSmiles).apply(Chem.MolToSmiles, canonical=True).values,
                                      augmented_scaffolds.values)

    def test_additional_basic_framework(self):
        # for i, row in self.others.iterrows():
        #     print(i,
        #           row.molecule_smiles,
        #           Chem.MolToSmiles(Chem.MolFromSmiles(row.basic_framework_smiles), canonical=True),
        #           Chem.MolToSmiles(scaffound.get_basic_framework(row.mol), canonical=True))
        #     self.assertTrue(Chem.MolToSmiles(Chem.MolFromSmiles(row.basic_framework_smiles), canonical=True) == Chem.MolToSmiles(scaffound.get_basic_framework(row.mol), canonical=True))
        basic_frameworks = self.others.mol.apply(scaffound.get_basic_framework).apply(Chem.MolToSmiles, canonical=True)
        np.testing.assert_array_equal(self.others.basic_framework_smiles.apply(Chem.MolFromSmiles).apply(Chem.MolToSmiles).values,
                                      basic_frameworks.values)

    def test_additional_basic_wireframe(self):
        # for i, row in self.others.iterrows():
        #     print(i,
        #           row.molecule_smiles,
        #           Chem.MolToSmiles(Chem.MolFromSmiles(row.basic_wireframe_smiles), canonical=True),
        #           Chem.MolToSmiles(scaffound.get_basic_wireframe(row.mol), canonical=True))
        #     self.assertTrue(Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(row.basic_wireframe_smiles), canonical=True)), canonical=True) == Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(scaffound.get_basic_wireframe(row.mol), canonical=True)), canonical=True))
        basic_wireframe = self.others.mol.apply(scaffound.get_basic_wireframe).apply(Chem.MolToSmiles, canonical=True)
        np.testing.assert_array_equal(self.others.basic_wireframe_smiles.apply(Chem.MolFromSmiles).apply(Chem.MolToSmiles, canonical=True).values,
                                      basic_wireframe.values)

    def test_additional_decorated_framework(self):
        # for i, row in self.others.iterrows():
        #     print(i, end=' ')
        #     print(row.molecule_smiles, end=' ')
        #     print(Chem.MolToSmiles(Chem.MolFromSmiles(row.decorated_framework_smiles), canonical=True), end=' ')
        #     print(Chem.MolToSmiles(scaffound.get_decorated_framework(row.mol), canonical=True))
        #     self.assertTrue(Chem.MolToSmiles(Chem.MolFromSmiles(row.decorated_framework_smiles), canonical=True) == Chem.MolToSmiles(scaffound.get_decorated_framework(row.mol), canonical=True))
        decorated_frameworks = self.others.mol.apply(scaffound.get_decorated_framework).apply(Chem.MolToSmiles, canonical=True)
        np.testing.assert_array_equal(self.others.decorated_framework_smiles.apply(Chem.MolFromSmiles).apply(Chem.MolToSmiles, canonical=True).values,
                                      decorated_frameworks.values)

    def test_additional_decorated_wireframe(self):
        # for i, row in self.others.iterrows():
        #     print(i,
        #           row.molecule_smiles,
        #           Chem.MolToSmiles(Chem.MolFromSmiles(row.decorated_wireframe_smiles), canonical=True),
        #           Chem.MolToSmiles(scaffound.get_decorated_wireframe(row.mol)))
        #     self.assertTrue(Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(row.decorated_wireframe_smiles), canonical=True)), canonical=True) == Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(scaffound.get_generic_graph(scaffound.get_saturated_graph(scaffound.get_decorated_scaffold(row.mol))), canonical=True)), canonical=True))
        decorated_wireframe = self.others.mol.apply(scaffound.get_decorated_wireframe).apply(Chem.MolToSmiles, canonical=True)
        np.testing.assert_array_equal(self.others.decorated_wireframe_smiles.apply(Chem.MolFromSmiles).apply(Chem.MolToSmiles, canonical=True).values,
                                      decorated_wireframe.values)

    def test_additional_augmented_framework(self):
        # for i, row in self.others.iterrows():
        #     print(i,
        #           row.molecule_smiles,
        #           Chem.MolToSmiles(Chem.MolFromSmiles(row.augmented_framework_smiles), canonical=True),
        #           Chem.MolToSmiles(scaffound.get_augmented_framework(row.mol, opts=self.opts), canonical=True))
        #     self.assertTrue(Chem.MolToSmiles(Chem.MolFromSmiles(row.augmented_framework_smiles), canonical=True) == Chem.MolToSmiles(scaffound.get_augmented_framework(row.mol, opts=self.opts), canonical=True))
        augmented_frameworks = self.others.mol.apply(scaffound.get_augmented_framework, opts=self.opts).apply(Chem.MolToSmiles, canonical=True)
        np.testing.assert_array_equal(self.others.augmented_framework_smiles.apply(Chem.MolFromSmiles).apply(Chem.MolToSmiles, canonical=True).values,
                                      augmented_frameworks.values)

    def test_additional_augmented_wireframe(self):
        # for i, row in self.others.iterrows():
        #     print(i,
        #           row.molecule_smiles,
        #           Chem.MolToSmiles(Chem.MolFromSmiles(row.augmented_wireframe_smiles), canonical=True),
        #           Chem.MolToSmiles(scaffound.get_augmented_wireframe(row.mol, opts=self.opts)))
        #     self.assertTrue(Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(row.augmented_wireframe_smiles), canonical=True)), canonical=True) == Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(scaffound.get_augmented_wireframe(row.mol, opts=self.opts), canonical=True)), canonical=True))
        augmented_wireframe = self.others.mol.apply(scaffound.get_augmented_wireframe, opts=self.opts).apply(Chem.MolToSmiles, canonical=True)
        np.testing.assert_array_equal(self.others.augmented_wireframe_smiles.apply(Chem.MolFromSmiles).apply(Chem.MolToSmiles, canonical=True).values,
                                      augmented_wireframe.values)

    def test_extended_molecular_anatomy(self):
        data = pd.concat([self.cox2, self.others], ignore_index=True)
        try:
            for mol in data.mol:
                anatomy = scaffound.MolecularAnatomy(mol)
                gg = anatomy.generic_graph
                scaffound.MolecularAnatomy(gg).as_table()
                sg = anatomy.saturated_graph
                scaffound.MolecularAnatomy(sg).as_table()
        except Exception as e:
            self.fail("MolecularAnatomy raised an exception: {}".format(e))

    def test_sanitization_warning(self):
        for i, mol in enumerate(chain(self.cox2.mol, self.others.mol)):
            bsc = scaffound.get_basic_scaffold(mol)
            sanitFail = Chem.SanitizeMol(bsc, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for BS for molecule {i}: {sanitFail}")
            dsc = scaffound.get_decorated_scaffold(mol)
            sanitFail = Chem.SanitizeMol(dsc, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for DS for molecule {i}: {sanitFail}")
            asc = scaffound.get_augmented_scaffold(mol, opts=self.opts)
            sanitFail = Chem.SanitizeMol(asc, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for AS for molecule {i}: {sanitFail}")
            bfr = scaffound.get_basic_framework(mol)
            sanitFail = Chem.SanitizeMol(bfr, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for BF for molecule {i}: {sanitFail}")
            dfr = scaffound.get_decorated_framework(mol)
            sanitFail = Chem.SanitizeMol(dfr, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for DF for molecule {i}: {sanitFail}")
            afr = scaffound.get_augmented_framework(mol, opts=self.opts)
            sanitFail = Chem.SanitizeMol(afr, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for AF for molecule {i}: {sanitFail}")
            bwf = scaffound.get_basic_wireframe(mol)
            sanitFail = Chem.SanitizeMol(bwf, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for BW for molecule {i}: {sanitFail}")
            dwf = scaffound.get_decorated_wireframe(mol)
            sanitFail = Chem.SanitizeMol(dwf, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for DW for molecule {i}: {sanitFail}")
            awf = scaffound.get_augmented_wireframe(mol, opts=self.opts)
            sanitFail = Chem.SanitizeMol(awf, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for AW for molecule {i}: {sanitFail}")

    def test_sanitization_warning_generic_graph_scaffolds(self):
        data = pd.concat([self.cox2, self.others], ignore_index=True)
        for i, mol in enumerate(data.mol):
            gg = scaffound.get_generic_graph(mol)
            sanitFail = Chem.SanitizeMol(gg, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for GG for molecule {i}: {sanitFail}")
            gbs = scaffound.get_basic_scaffold(gg)
            sanitFail = Chem.SanitizeMol(gbs, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for GBS for molecule {i}: {sanitFail}")
            gds = scaffound.get_decorated_scaffold(gg)
            sanitFail = Chem.SanitizeMol(gds, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for GDS for molecule {i}: {sanitFail}")
            gas = scaffound.get_augmented_scaffold(gg, opts=self.opts)
            sanitFail = Chem.SanitizeMol(gas, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for GAS for molecule {i}: {sanitFail}")
            gbf = scaffound.get_generic_graph(gbs)
            sanitFail = Chem.SanitizeMol(gbf, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for GBF for molecule {i}: {sanitFail}")
            gdf = scaffound.get_generic_graph(gds)
            sanitFail = Chem.SanitizeMol(gdf, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for GDF for molecule {i}: {sanitFail}")
            gaf = scaffound.get_generic_graph(gas)
            sanitFail = Chem.SanitizeMol(gaf, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for GAF for molecule {i}: {sanitFail}")
            gbw = scaffound.get_generic_graph(scaffound.get_saturated_graph(gbs))
            sanitFail = Chem.SanitizeMol(gbw, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for GBW for molecule {i}: {sanitFail}")
            gdw = scaffound.get_generic_graph(scaffound.get_saturated_graph(gds))
            sanitFail = Chem.SanitizeMol(gdw, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for GDW for molecule {i}: {sanitFail}")
            gaw = scaffound.get_generic_graph(scaffound.get_saturated_graph(gas))
            sanitFail = Chem.SanitizeMol(gaw, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for GAW for molecule {i}: {sanitFail}")

    def test_sanitization_warning_saturated_graph_scaffolds(self):
        data = pd.concat([self.cox2, self.others], ignore_index=True)
        for i, mol in enumerate(data.mol):
            gg = scaffound.get_saturated_graph(mol)
            sanitFail = Chem.SanitizeMol(gg, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for GG for molecule {i}: {sanitFail}")
            gbs = scaffound.get_basic_scaffold(gg)
            sanitFail = Chem.SanitizeMol(gbs, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for GBS for molecule {i}: {sanitFail}")
            gds = scaffound.get_decorated_scaffold(gg)
            sanitFail = Chem.SanitizeMol(gds, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for GDS for molecule {i}: {sanitFail}")
            gas = scaffound.get_augmented_scaffold(gg, opts=self.opts)
            sanitFail = Chem.SanitizeMol(gas, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for GAS for molecule {i}: {sanitFail}")
            gbf = scaffound.get_generic_graph(gbs)
            sanitFail = Chem.SanitizeMol(gbf, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for GBF for molecule {i}: {sanitFail}")
            gdf = scaffound.get_generic_graph(gds)
            sanitFail = Chem.SanitizeMol(gdf, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for GDF for molecule {i}: {sanitFail}")
            gaf = scaffound.get_generic_graph(gas)
            sanitFail = Chem.SanitizeMol(gaf, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for GAF for molecule {i}: {sanitFail}")
            gbw = scaffound.get_generic_graph(scaffound.get_saturated_graph(gbs))
            sanitFail = Chem.SanitizeMol(gbw, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for GBW for molecule {i}: {sanitFail}")
            gdw = scaffound.get_generic_graph(scaffound.get_saturated_graph(gds))
            sanitFail = Chem.SanitizeMol(gdw, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for GDW for molecule {i}: {sanitFail}")
            gaw = scaffound.get_generic_graph(scaffound.get_saturated_graph(gas))
            sanitFail = Chem.SanitizeMol(gaw, catchErrors=True)
            if sanitFail:
                raise ValueError(f"SanitizeMol raised an exception for GAW for molecule {i}: {sanitFail}")

if __name__ == '__main__':
    unittest.main()
