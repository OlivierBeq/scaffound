import unittest
from itertools import chain

from rdkit import Chem

import scaffound


class LongestPathTestCase(unittest.TestCase):

    def setUp(self):
        smiles_path = {
            # Order of heteroatoms
            'C(NOSC1=CC=C(C(C2=CC=C(NSOCO)C=C2)C)C=C1)O': ('unique path with max atomic number topology', [20, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
            # Exact equivalence
            'C(O)C1C=CC(C2C=CC(C3C=CC(CC(CC4C=CC(C5C=CC(C6C=CC(CO)=CC=6)=CC=5)=CC=4)C)=CC=3)=CC=2)=CC=1': ('paths are equivalent', None),
            # Heaviest heteroatom
            'C1C=C(C=CC=1COCO)C1C=CC(C2C=CC(CC(CC3C=CC(C4C=CC(C5C=CC(CNCO)=CC=5)=CC=4)=CC=3)C)=CC=2)=CC=1': ('unique path with max total atomic number', [43, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 2, 1, 0, 5, 6, 7, 8]),
            # Order of bond order (with aromatic rings)
            'C(O)C1CCC(C2C=CC(C3CCC(CC(CC4=CC=C(C5CCC(C6CCC(CO)CC6)CC5)C=C4)C)CC3)=CC=2)CC1': ('unique path with max bond order topology', [37, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]),
            # Order of bond order (with aromatic rings)
            'C(O)C1CCC(C2C=CC(C3CCC(CC(CC4CCC(C5CCC(C6=CC=C(CO)C=C6)CC5)CC4)C)CC3)=CC=2)CC1': ('unique path with max bond order topology', [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 37]),
            # Priority of aromatic rings
            'C1(C=CC(CONCO)=CC=1)C(C)C1CCC(CONCO)CC1': ('unique path with max aromatic rings', [12, 11, 0, 1, 2, 3, 4, 5, 6, 7]),
            # Order of heteroatoms
            'C1(C(C2=CC=C(CNOCO)C=C2)C)=CC=C(CONCO)C=C1': ('unique path with max atomic number topology', [13, 1, 0, 14, 15, 16, 17, 18, 19, 20]),
            # Priority of unsaturated bonds
            r'CC(/C=C/C1CCC(CO)CC1)CCC1CCC(CO)CC1': ('unique path with max bond order topology', [0, 1, 2, 3, 4, 5, 6, 7, 8]),
            # Minimum number of ring atoms
            'CCCC1C=C(C2CCC(CCO)CC2)C(OC2CCCC(CO)C2)=CC=1': ('unique path with max total atomic number', [0, 1, 2, 3, 4, 5, 15, 16, 17, 24, 21, 22]),
            # Minimum number of ring atoms
            'CCCC1=CC=C(CC2CC(CO)CCC2)C(C2CCCC(CCO)C2)=C1': ('unique path with min ring atoms', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            # Minimum number of ring atoms
            'CCCC1=CC=C(CC2CC(CO)CCC2)C(C2CCCC(CCCO)C2)=C1': ('unique path with min ring atoms', [0, 1, 2, 3, 27, 16, 17, 26, 21, 22, 23, 24]),
            # Minimum number of ring atoms
            'CCCC1=CC=C(OC2CC(CO)CCC2)C(OC2CCCC(CO)C2)=C1': ('unique path with min ring atoms', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            # Priority of smaller rings
            'CC(C1CCC(C2CC(CO)CC2)CC1)C1CCCC(C2CC(CO)CC2)CC1': ('unique path with min ring size', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            # Priority to heteroatoms
            'C1C(COC(C)=O)C(COC(C)=O)CC(CCCC)N1CCCC': ('unique path with max atomic number topology', [11, 10, 9, 8, 7, 1, 0, 19, 20, 21, 22, 23]),
            # Exact equivalence
            'C1C(COC(C)=O)C(COC(C)=O)CC(CCCC)C1CCCC': ('paths are equivalent', None),
        }
        self.mols = list(map(Chem.MolFromSmiles, smiles_path.keys()))
        self.debug_result, self.lsp = tuple(zip(*smiles_path.values()))
        self.opts = scaffound.paths.MinMaxShortestPathOptions(
            original_algorithm=False,
            select_path_len = scaffound.paths.SelectionMethod.MAXIMIZE,
            select_ring_count = scaffound.paths.SelectionMethod.MAXIMIZE,
            select_ring_size = scaffound.paths.SelectionMethod.MINIMIZE,
            select_arom_rings = scaffound.paths.SelectionMethod.MAXIMIZE,
            select_assymetry = scaffound.paths.SelectionMethod.MAXIMIZE,
            select_num_ring_atoms = scaffound.paths.SelectionMethod.MINIMIZE,
            select_total_atomic_num = scaffound.paths.SelectionMethod.MAXIMIZE,
            select_isotopes = scaffound.paths.SelectionMethod.MAXIMIZE,
            select_atom_num_topology_dir = scaffound.paths.SelectionMethod.MINIMIZE,
            select_atom_num_topology = scaffound.paths.SelectionMethod.MAXIMIZE,
            select_bond_order_topology_dir = scaffound.paths.SelectionMethod.MAXIMIZE,
            select_bond_order_topology = scaffound.paths.SelectionMethod.MAXIMIZE,
            debug = True)

    def test_longest_path(self):
        for i, mol in enumerate(self.mols):
            bsc_atoms = scaffound.get_basic_scaffold(mol, only_atom_indices=True)
            dsc_atoms = scaffound.get_decorated_scaffold(mol, only_atom_indices=True)
            # Find true terminal carbon atoms
            true_terminal_carbons = list(chain.from_iterable(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6&D1]'))))
            # Find bespoke terminal carbon atoms
            bespoke_terminal_carbons = [atom.GetIdx() for atom in mol.GetAtoms()
                                        if atom.GetIdx() not in dsc_atoms and atom.GetSymbol() == 'C']
            # Only consider bespoke terminal carbons if no true terminal carbons are found
            if len(true_terminal_carbons) > 0:
                # Include bespoke terminal carbons as possible endpoints only if they are not part of the longest shortest path
                longest_shortest_path = scaffound.paths.get_min_max_shortest_path_without_symmetry(mol,
                                                                                                   true_terminal_carbons,
                                                                                                   bsc_atoms,
                                                                                                   opts=self.opts
                                                                                                   )
                terminal_carbons = true_terminal_carbons
                possible_endpoints = list(set(bsc_atoms) | set(atom
                                                               for atom in bespoke_terminal_carbons
                                                               if atom not in longest_shortest_path
                                                               )
                                          )
            elif len(bespoke_terminal_carbons) == 1:
                terminal_carbons = bespoke_terminal_carbons
                possible_endpoints = bsc_atoms
            else:
                # Consider bespoke terminal carbons only if they are not part of the longest shortest path
                longest_shortest_path = scaffound.paths.get_min_max_shortest_path_without_symmetry(mol,
                                                                                                   bespoke_terminal_carbons,
                                                                                                   bsc_atoms,
                                                                                                   opts=self.opts
                                                                                                   )
                terminal_carbons = [atom
                                    for atom in bespoke_terminal_carbons
                                    if atom not in longest_shortest_path or atom in [longest_shortest_path[0],
                                                                                     longest_shortest_path[-1]]]
                possible_endpoints = bsc_atoms
            # Find the longest shortest path between terminal carbons
            lsp, debug_info = scaffound.paths.get_min_max_shortest_path_without_symmetry(mol, terminal_carbons,
                                                                                         possible_endpoints, opts=self.opts
                                                                                         )
            print(i, 'FINAL', debug_info.get('result'), Chem.MolToSmiles(mol, canonical=False), lsp,
                  ''.join([mol.GetAtomWithIdx(idx).GetSymbol() for idx in lsp])
                  )
            self.assertEqual(self.debug_result[i], debug_info.get('result'))
            if debug_info.get('result') != 'paths are equivalent':
                self.assertEqual(sorted(self.lsp[i]), sorted(lsp))

if __name__ == '__main__':
    unittest.main()
