# -*- coding: utf-8 -*-


"""Implementations of shortest and longest paths."""

from enum import Enum, auto as enum_auto
from collections import deque, OrderedDict
from itertools import combinations, product

from rdkit import Chem

from . import scaffolds


# Convenience fn
str_fn = lambda x: {max: 'max', min: 'min'}.get(x)

class SelectionMethod(Enum):
    MINIMIZE = enum_auto()
    MAXIMIZE = enum_auto()

class MinMaxShortestPathOptions:

    def __init__(self, original_algorithm: bool = False,
                 select_path_len: SelectionMethod = SelectionMethod.MAXIMIZE,
                 select_ring_count: SelectionMethod = SelectionMethod.MAXIMIZE,
                 select_ring_size: SelectionMethod = SelectionMethod.MINIMIZE,
                 select_arom_rings: SelectionMethod = SelectionMethod.MAXIMIZE,
                 select_assymetry: SelectionMethod = SelectionMethod.MAXIMIZE,
                 select_num_ring_atoms: SelectionMethod = SelectionMethod.MINIMIZE,
                 select_total_atomic_num: SelectionMethod = SelectionMethod.MAXIMIZE,
                 select_isotopes: SelectionMethod = SelectionMethod.MAXIMIZE,
                 select_atom_num_topology: SelectionMethod = SelectionMethod.MAXIMIZE,
                 select_atom_num_topology_dir: SelectionMethod = SelectionMethod.MINIMIZE,
                 select_bond_order_topology: SelectionMethod = SelectionMethod.MAXIMIZE,
                 select_bond_order_topology_dir: SelectionMethod = SelectionMethod.MAXIMIZE,
                 debug: bool = False):
        """Setup selection criteria for the min/max shortest path algorithm, to disambiguate from paths with same lengths.

        :param original_algorithm: If true, the disambiguation process does not occur.
        The first longest path identified is returned. This corresponds to the authors' reported algorithm.
        :param select_path_len: Whether to identify the shortest (MINIMIZE) or longest (MAXIMIZE) path
        :param select_ring_count: Whether to prioritize paths with the least (MINIMIZE) or most (MAXIMIZE) rings
        :param select_ring_size: Whether to prioritize paths with the smallest (MINIMIZE) or largest MAXIMUM) rings
        :param select_arom_rings: Whether to prioritize paths with the least (MINIMIZE) or most (MAXIMIZE) aromatic rings
        :param select_assymetry: Whether to prioritize symmetrical (MINIMIZE) or asymmetrical (MAXIMIZE) paths
        :param select_num_ring_atoms: Whether to prioritize paths with the least (MINIMIZE) or most (MAXIMIZE) ring atoms
        :param select_total_atomic_num: Whether to prioritize paths whose sum of atomic number of their atoms is the smallest (MINIMIZE) or largest (MAXIMIZE)
        :param select_isotopes: Whether to prioritize paths whose sum of masses of atypical isotopes is the smallest (MINIMIZE) or largest (MAXIMIZE)
        :param select_atom_num_topology: Whether to prioritize paths with smaller (MINIMIZE) or larger (MAXIMIZE) atomic number sequences.
        This is determined by alphanumerically comparing the sequences of atomic numbers of the atoms in each path.
        :param select_atom_num_topology_dir: For atomic number sequences, whether to consider the forward (MINIMIZE) or reverse (MAXIMIZE) order of the sequence.
        :param select_bond_order_topology: Whether to prioritize paths with smaller (MINIMIZE) or larger (MAXIMIZE) bond order sequences.
        This is determined by alphanumerically comparing the sequences of bond orders in each path.
        :param select_bond_order_topology_dir: For bond order sequences, whether to consider the forward (MINIMIZE) or reverse (MAXIMIZE) order of the sequence.
        :param debug: If True, debug information about dropped paths are also returned.
        """
        self.original_algorithm = original_algorithm
        self.select_path_len = max if select_path_len == SelectionMethod.MAXIMIZE else min
        self.select_ring_count = max if select_ring_count == SelectionMethod.MAXIMIZE else min
        self.select_ring_size = max if select_ring_size == SelectionMethod.MAXIMIZE else min
        self.select_arom_rings = max if select_arom_rings == SelectionMethod.MAXIMIZE else min
        self.select_assymetry = max if select_assymetry == SelectionMethod.MAXIMIZE else min
        self.select_num_ring_atoms = max if select_num_ring_atoms == SelectionMethod.MAXIMIZE else min
        self.select_total_atomic_num = max if select_total_atomic_num == SelectionMethod.MAXIMIZE else min
        self.select_isotopes = max if select_isotopes == SelectionMethod.MAXIMIZE else min
        self.select_atom_num_topology_dir = max if select_atom_num_topology_dir == SelectionMethod.MAXIMIZE else min
        self.select_atom_num_topology = max if select_atom_num_topology == SelectionMethod.MAXIMIZE else min
        self.select_bond_order_topology_dir = max if select_bond_order_topology_dir == SelectionMethod.MAXIMIZE else min
        self.select_bond_order_topology = max if select_bond_order_topology == SelectionMethod.MAXIMIZE else min
        self.debug = debug


def get_min_max_shortest_path_without_symmetry(
        mol: Chem.Mol,
        indices: list[int],
        basic_scaffold: list[int] = None,
        opts: MinMaxShortestPathOptions = None) -> list[int] | tuple[list[int], dict]:
    """
    Find the longest/shortest path between any two points of a list of atom indices,
    with a tie-breaking rule based on a basic scaffold.

    When multiple paths have the same longest/shortest length, it prefers paths that,
    after removing atoms belonging to the basic_scaffold, result in chemically
    unique fragments.

    :param mol: The molecule to search within.
    :param possible_endpoints: List of candidate atom indices for the endpoint.
    :param basic_scaffold: A list of atom indices representing a scaffold. Used for tie-breaking. Defaults to empty.
    :param opts: Options for each of the selection steps.
    :return: A list of atom indices for the chosen shortest path, together with debugging information if opts.debug is True.
    """
    if not isinstance(mol, Chem.Mol):
        raise ValueError("Molecule must be a valid RDKit Chem.Mol.")
    if not isinstance(indices, list):
        raise ValueError("indices must be a valid list.")
    if not isinstance(opts, MinMaxShortestPathOptions) and opts is not None:
        raise ValueError("opts must be a valid MinMaxShortestPathOptions.")
    # Ensure aromaticity is perceived
    mol = Chem.Mol(mol) # copy
    Chem.SanitizeMol(mol)
    if opts is None:
        opts = MinMaxShortestPathOptions()
    if opts.original_algorithm:
        return get_original_min_max_shortest_path_without_symmetry(mol=mol, indices=indices,
                                                                   basic_scaffold=basic_scaffold,
                                                                   opts=opts)
    if opts.debug:
        debug_info = OrderedDict({'input_smiles': Chem.MolToSmiles(mol, canonical=False)})
    scaffold_set = set(basic_scaffold) or set()
    # Find all paths and their lengths
    all_paths = []
    for start, end in combinations(indices, 2):
        # Ensure start and end are not the same
        path = Chem.GetShortestPath(mol, start, end)
        if path and not scaffold_set.isdisjoint(path):
            all_paths.append(list(path))
    if opts.debug:
        debug_info["all_paths_terminal_atoms"] = all_paths
    if not all_paths:
        # No path between 2 terminal carbon atoms exists
        # Retry with the basic scaffold
        for start, end in product(indices, scaffold_set):
            if start == end:
                continue
            # Ensure start and end are not the same
            path = Chem.GetShortestPath(mol, start, end)
            if path:
                all_paths.append(list(path))
        if not all_paths:
            # Cannot find a path
            if opts.debug:
                debug_info["result"] = 'no path found'
                return [], debug_info
            return []
        if opts.debug:
            debug_info["all_paths_terminal_and_scaffold_atoms"] = all_paths
    # Identify all paths with the maximum/minimum length
    minmax_len = opts.select_path_len(map(len, all_paths))
    shortest_paths_group = [p for p in all_paths if len(p) == minmax_len]
    if opts.debug:
        debug_info[f'{str_fn(opts.select_path_len)}_path_len'] = minmax_len
        debug_info[f'paths_with_{str_fn(opts.select_path_len)}_len'] = shortest_paths_group
        debug_info[f'paths_without_{str_fn(opts.select_path_len)}_len'] = [p for p in all_paths if len(p) != minmax_len]
    # No tie-break needed if there's only one shortest path
    if len(shortest_paths_group) <= 1:
        if opts.debug:
            debug_info["result"] = (f'unique {str_fn(opts.select_path_len)} path found'
                                    if len(shortest_paths_group)
                                    else 'no path found')
            return (shortest_paths_group[0] if shortest_paths_group else []), debug_info
        return shortest_paths_group[0] if shortest_paths_group else []
    # Map each atom to the rings it belongs to
    ring_info = mol.GetRingInfo()
    atom_to_rings_map = [[] for _ in range(mol.GetNumAtoms())]
    ringsize_map = {}
    ring_aromat_map = {}
    for ring_idx, ring_atoms in enumerate(ring_info.AtomRings()):
        for atom_idx in ring_atoms:
            atom_to_rings_map[atom_idx].append(ring_idx)
        ringsize_map[ring_idx] = len(ring_atoms)
        ring_aromat_map[ring_idx] = all(mol.GetAtomWithIdx(atom).GetIsAromatic() for atom in ring_atoms)
    # Find paths that go through the minimum number of rings
    paths_with_ring_counts = [] # number of rings
    paths_with_ringsize = [] # size of rings
    paths_with_ring_aromat_count = [] # are rings aromatic
    for path in shortest_paths_group:
        visited_rings = set()
        for atom_idx in path:
            # Add all rings this atom belongs to into the set
            visited_rings.update(atom_to_rings_map[atom_idx])
        paths_with_ring_counts.append((path, len(visited_rings)))
        # Get the size of each ring in the set of unique rings in this path
        paths_with_ringsize.append((path, sum(ringsize_map[ring_idx] for ring_idx in visited_rings)))
        # Get the number of aromatic rings
        paths_with_ring_aromat_count.append((path, sum(ring_aromat_map[ring_idx] for ring_idx in visited_rings)))
    min_visited_rings = opts.select_ring_count(count for path, count in paths_with_ring_counts)
    paths_min_rings = [path for path, count in paths_with_ring_counts if count == min_visited_rings]
    if opts.debug:
        debug_info[f'{str_fn(opts.select_ring_count)}_ring_count'] = min_visited_rings
        debug_info[f'paths_with_{str_fn(opts.select_ring_count)}_rings'] = paths_min_rings
        debug_info[f'paths_without_{str_fn(opts.select_ring_count)}_rings'] = [path for path, count in paths_with_ring_counts if count != min_visited_rings]
    # No further tie-break needed if there's only one shortest path
    if len(paths_min_rings) == 1:
        if opts.debug:
            debug_info["result"] = f'unique path with {str_fn(opts.select_ring_count)} rings found'
            return paths_min_rings[0], debug_info
        return paths_min_rings[0]
    # Prioritize paths with smaller rings
    min_total_ringsize = opts.select_ring_size(total_ringsize
                                               for path, total_ringsize in paths_with_ringsize
                                               if path in paths_min_rings)
    paths_min_ringsize = [path for path, total_ringsize in paths_with_ringsize
                          if path in paths_min_rings and total_ringsize == min_total_ringsize]
    if opts.debug:
        debug_info[f'{str_fn(opts.select_ring_size)}_ring_size'] = min_total_ringsize
        debug_info[f'paths_with_{str_fn(opts.select_ring_size)}_ring_size'] = paths_min_ringsize
        debug_info[f'paths_without_{str_fn(opts.select_ring_size)}_ring_size'] = [path
                                                                                  for (path, total_ringsize), (_, count) in zip(paths_with_ringsize, paths_with_ring_counts)
                                                                                  if count == min_visited_rings and total_ringsize != min_total_ringsize]
    if len(paths_min_ringsize) == 1:
        if opts.debug:
            debug_info["result"] = f'unique path with {str_fn(opts.select_ring_size)} ring size'
            return paths_min_ringsize[0], debug_info
        return paths_min_ringsize[0]
    # Prioritize paths with aromatic rings
    max_arom_rings = opts.select_arom_rings(count for path, count in paths_with_ring_aromat_count if path in paths_min_ringsize)
    paths_max_arom_rings = [path for path, count in paths_with_ring_aromat_count if path in paths_min_ringsize and count == max_arom_rings]
    if opts.debug:
        debug_info[f'{str_fn(opts.select_arom_rings)}_aromatic_rings'] = max_arom_rings
        debug_info[f'paths_with_{str_fn(opts.select_arom_rings)}_aromatic_rings'] = paths_max_arom_rings
        debug_info[f'paths_without_{str_fn(opts.select_arom_rings)}_aromatic_rings'] = [path
                                                                                        for path, count in paths_with_ring_aromat_count
                                                                                        if count != max_arom_rings]
    if len(paths_max_arom_rings) == 1:
        if opts.debug:
            debug_info["result"] = f'unique path with {str_fn(opts.select_arom_rings)} aromatic rings'
            return paths_max_arom_rings[0], debug_info
        return paths_max_arom_rings[0]
    # Favour paths with the least total number of ring atoms
    paths_with_ring_atom_counts = []
    for path in paths_max_arom_rings:
        ring_atom_count = sum(1 for idx in path if mol.GetAtomWithIdx(idx).IsInRing())
        paths_with_ring_atom_counts.append((path, ring_atom_count))
    # Filter for paths having the minimum number of ring atoms
    min_ring_atom_count = opts.select_num_ring_atoms(count for path, count in paths_with_ring_atom_counts)
    paths_min_ring_atoms = [path for path, count in paths_with_ring_atom_counts if count == min_ring_atom_count]
    if opts.debug:
        debug_info[f'{str_fn(opts.select_num_ring_atoms)}_ring_atoms'] = min_ring_atom_count
        debug_info[f'paths_with_{str_fn(opts.select_num_ring_atoms)}_ring_atoms'] = paths_min_ring_atoms
        debug_info[f'paths_without_{str_fn(opts.select_num_ring_atoms)}_ring_atoms'] = [path for path, count in paths_with_ring_atom_counts if count != min_ring_atom_count]
    if len(paths_min_ring_atoms) == 1:
        if opts.debug:
            debug_info["result"] = f'unique path with {str_fn(opts.select_num_ring_atoms)} ring atoms'
            return paths_min_ring_atoms[0], debug_info
        return paths_min_ring_atoms[0]
    # Favour paths with asymmetry
    true_bsc_atoms = set(scaffolds.get_basic_scaffold(mol, only_atom_indices=True))
    path_fragment_smiles = [] # Canonical SMILES for the fragments of each path
    for path in paths_min_ring_atoms:
        # Determine the atoms in the fragment (path atoms minus scaffold atoms)
        fragment_indices = sorted(list(set(path) - true_bsc_atoms))
        if fragment_indices:
            # Create a new molecule from the fragment indices
            rw_mol = Chem.RWMol(mol)
            # Remove all atoms NOT in our fragment
            atoms_to_remove = [i for i in range(mol.GetNumAtoms()) if i not in fragment_indices]
            scaffolds.unassign_chirality_and_delete(rw_mol, atoms_to_remove)
            # Get fragments
            fragments = Chem.GetMolFrags(rw_mol.GetMol(), asMols=True)
            # Use canonical SMILES to uniquely identify the fragment's structure
            fragment_smiles = map(Chem.MolToSmiles, fragments)
            path_fragment_smiles.append((path, set(fragment_smiles)))
    # Select the path avoiding unique fragments
    max_fragments = opts.select_assymetry(map(len, list(zip(*path_fragment_smiles))[1]))
    path_max_fragments = [path for path, frags in path_fragment_smiles if len(frags) == max_fragments]
    if opts.debug:
        debug_info[f'path_fragments_for_asymmetry'] = path_fragment_smiles
        debug_info[f'{str_fn(opts.select_assymetry)}_asymmetry_fragments'] = max_fragments
        debug_info[f'paths_with_{str_fn(opts.select_assymetry)}_asymmetry_fragments'] = path_max_fragments
        debug_info[f'paths_without_{str_fn(opts.select_assymetry)}_asymmetry_fragments'] = [path for path, frags in path_fragment_smiles if len(frags) != max_fragments]
    if len(path_max_fragments) == 1:
        if opts.debug:
            debug_info["result"] = f'unique path with {str_fn(opts.select_assymetry)} fragment asymmetry'
            return path_max_fragments[0], debug_info
        return path_max_fragments[0]
    # Prioritize paths with the maximum cumulated sum of atomic numbers
    paths_total_atomnum = []
    for path in path_max_fragments:
        total_atomnum = sum(mol.GetAtomWithIdx(idx).GetAtomicNum() for idx in path)
        paths_total_atomnum.append((path, total_atomnum))
        # Filter for paths having the minimum number of ring atoms
    max_atomnum = opts.select_total_atomic_num(total for path, total in paths_total_atomnum)
    paths_max_atomnum = [path for path, total in paths_total_atomnum if total == max_atomnum]
    if opts.debug:
        debug_info[f'{str_fn(opts.select_total_atomic_num)}_total_atomic_num'] = max_atomnum
        debug_info[f'paths_with_{str_fn(opts.select_total_atomic_num)}_total_atomic_num'] = paths_max_atomnum
        debug_info[f'paths_without_{str_fn(opts.select_total_atomic_num)}_total_atomic_num'] = [path for path, total in paths_total_atomnum if total != max_atomnum]
    if len(paths_max_atomnum) == 1:
        if opts.debug:
            debug_info["result"] = f'unique path with {str_fn(opts.select_total_atomic_num)} total atomic number'
            return paths_max_atomnum[0], debug_info
        return paths_max_atomnum[0]
    # Prioritize paths with atypical isotopes
    periodic_table = Chem.GetPeriodicTable()
    paths_isotopes = []
    for path in paths_max_atomnum:
        total_atypical_isotopes = sum(mol.GetAtomWithIdx(idx).GetIsotope()
                                      for idx in path
                                      if periodic_table.GetMostCommonIsotope(mol.GetAtomWithIdx(idx).GetAtomicNum()) != mol.GetAtomWithIdx(idx).GetIsotope())
        paths_isotopes.append((path, total_atypical_isotopes))
    max_atypical_isotopes = opts.select_isotopes(total for path, total in paths_isotopes)
    paths_max_isotopes = [path for path, total in paths_isotopes if total == max_atypical_isotopes]
    if opts.debug:
        debug_info[f'{str_fn(opts.select_isotopes)}_total_atypical_isotope'] = max_atypical_isotopes
        debug_info[f'paths_with_{str_fn(opts.select_isotopes)}_total_atypical_isotope'] = paths_max_isotopes
        debug_info[f'paths_without_{str_fn(opts.select_isotopes)}_total_atypical_isotope'] = [path for path, total in paths_isotopes if total != max_atypical_isotopes]
    if len(paths_max_isotopes) == 1:
        if opts.debug:
            debug_info["result"] = f'unique path with {str_fn(opts.select_isotopes)} total atypical isotopic mass'
            return paths_max_isotopes[0], debug_info
        return paths_max_isotopes[0]
    # Prioritize paths whose sequence of atomic numbers grows the most and the fastest
    paths_with_seq_atomnum = []
    for path in paths_max_isotopes:
        seq_atomnum = [mol.GetAtomWithIdx(idx).GetAtomicNum() for idx in path]
        # Consider the reverse orientation of the sequence
        seq_atomnum = opts.select_atom_num_topology_dir(seq_atomnum, seq_atomnum[::-1])
        paths_with_seq_atomnum.append((path, seq_atomnum))
    max_seq_atomnum = opts.select_atom_num_topology(seq_atomnum for path, seq_atomnum in paths_with_seq_atomnum)
    paths_min_seq_atomnum = [path for path, seq_atomnum in paths_with_seq_atomnum if seq_atomnum == max_seq_atomnum]
    if opts.debug:
        debug_info[f'{str_fn(opts.select_atom_num_topology)}_topology_atomic_nums'] = max_seq_atomnum
        debug_info[f'paths_with_{str_fn(opts.select_atom_num_topology)}_topology_atomic_nums'] = paths_min_seq_atomnum
        debug_info[f'paths_without_{str_fn(opts.select_atom_num_topology)}_topology_atomic_nums'] = [path for path, seq_atomnum in paths_with_seq_atomnum if seq_atomnum != max_seq_atomnum]
    if len(paths_min_seq_atomnum) == 1:
        if opts.debug:
            debug_info["result"] = f'unique path with {str_fn(opts.select_atom_num_topology)} atomic number topology'
            return paths_min_seq_atomnum[0], debug_info
        return paths_min_seq_atomnum[0]
    # Prioritize paths whose sequence of bond orders grows the fastest
    paths_with_seq_bondorder = []
    for path in paths_min_seq_atomnum:
        seq_bondorder = [mol.GetBondBetweenAtoms(path[i - 1], path[i]).GetBondTypeAsDouble()
                         for i in range(1, len(path))]
        # Consider the reverse orientation of the sequence
        seq_bondorder = opts.select_bond_order_topology_dir(seq_bondorder, seq_bondorder[::-1])
        paths_with_seq_bondorder.append((path, seq_bondorder))
    max_seq_bond_order = opts.select_bond_order_topology(seq_bondorder for path, seq_bondorder in paths_with_seq_bondorder)
    paths_min_seq_bondorder = [path for path, seq_bondorder in paths_with_seq_bondorder if seq_bondorder == max_seq_bond_order]
    if opts.debug:
        debug_info[f'{str_fn(opts.select_bond_order_topology)}_topology_bond_orders'] = max_seq_bond_order
        debug_info[f'paths_with_{str_fn(opts.select_bond_order_topology)}_topology_bond_orders'] = paths_min_seq_bondorder
        debug_info[f'paths_without_{str_fn(opts.select_bond_order_topology)}_topology_bond_orders'] = [path for path, seq_bondorder in paths_with_seq_bondorder if seq_bondorder != max_seq_bond_order]
    if len(paths_min_seq_bondorder) == 1:
        if opts.debug:
            debug_info["result"] = f'unique path with {str_fn(opts.select_bond_order_topology)} bond order topology'
            return paths_min_seq_bondorder[0], debug_info
        return paths_min_seq_bondorder[0]
    # All paths are symmetrically identical, choose one
    if opts.debug:
        debug_info["result"] = f'paths are equivalent'
        return paths_min_seq_bondorder[0], debug_info
    return paths_min_seq_bondorder[0]

def get_original_min_max_shortest_path_without_symmetry(
        mol: Chem.Mol,
        indices: list[int],
        basic_scaffold: list[int] = None,
        opts: MinMaxShortestPathOptions = None) -> list[int] | tuple[list[int], dict]:
    """
    Find the longest/shortest path between any two points of a list of atom indices,
    with only one tie-breaking rule.

    When multiple paths have the same longest/shortest length, paths that,
    after removing atoms belonging to the basic_scaffold, result in chemically
    unique fragments are preferred.

    :param mol: The molecule to search within.
    :param possible_endpoints: List of candidate atom indices for the endpoint.
    :param basic_scaffold: A list of atom indices representing a scaffold. Used for tie-breaking. Defaults to empty.
    :param opts: Options for each of the selection steps.
    :return: A list of atom indices for the chosen shortest path, together with debugging information if opts.debug is True.
    """
    if not isinstance(mol, Chem.Mol):
        raise ValueError("Molecule must be a valid RDKit Chem.Mol.")
    if not isinstance(indices, list):
        raise ValueError("indices must be a valid list.")
    if opts.debug:
        debug_info = OrderedDict({'input_smiles': Chem.MolToSmiles(mol, canonical=False)})
    scaffold_set = set(basic_scaffold) or set()
    # Find all paths and their lengths
    all_paths = []
    for start, end in combinations(indices, 2):
        # Ensure start and end are not the same
        path = Chem.GetShortestPath(mol, start, end)
        if path and not scaffold_set.isdisjoint(path):
            all_paths.append(list(path))
    if opts.debug:
        debug_info["all_paths_terminal_atoms"] = all_paths
    if not all_paths:
        # No path between 2 terminal carbon atoms exists
        # Retry with the basic scaffold
        for start, end in product(indices, scaffold_set):
            if start == end:
                continue
            # Ensure start and end are not the same
            path = Chem.GetShortestPath(mol, start, end)
            if path:
                all_paths.append(list(path))
        if not all_paths:
            # Cannot find a path
            if opts.debug:
                debug_info["result"] = 'no path found'
                return [], debug_info
            return []
        if opts.debug:
            debug_info["all_paths_terminal_and_scaffold_atoms"] = all_paths
    # Identify all paths with the minimum length
    minmax_len = opts.select_path_len(map(len, all_paths))
    shortest_paths_group = [p for p in all_paths if len(p) == minmax_len]
    if opts.debug:
        debug_info[f'{str_fn(opts.select_path_len)}_path_len'] = minmax_len
        debug_info[f'paths_with_{str_fn(opts.select_path_len)}_len'] = shortest_paths_group
        debug_info[f'paths_without_{str_fn(opts.select_path_len)}_len'] = [p for p in all_paths if len(p) != minmax_len]
    # No tie-break needed if there's only one shortest path
    if len(shortest_paths_group) <= 1:
        if opts.debug:
            debug_info["result"] = (f'unique {str_fn(opts.select_path_len)} path found'
                                    if len(shortest_paths_group)
                                    else 'no path found')
            return (shortest_paths_group[0] if shortest_paths_group else []), debug_info
        return shortest_paths_group[0] if shortest_paths_group else []
    # Tie-Breaking Logic
    path_fragment_smiles = [] # Canonical SMILES for the fragments of each path
    for path in shortest_paths_group:
        # Determine the atoms in the fragment (path atoms minus scaffold atoms)
        fragment_indices = sorted(list(set(path) - scaffold_set))
        if fragment_indices:
            # Create a new molecule from the fragment indices
            rw_mol = Chem.RWMol(mol)
            # Remove all atoms NOT in our fragment
            atoms_to_remove = [i for i in range(mol.GetNumAtoms()) if i not in fragment_indices]
            scaffolds.unassign_chirality_and_delete(rw_mol, atoms_to_remove)
            # Get fragments
            fragments = Chem.GetMolFrags(rw_mol.GetMol(), asMols=True)
            # Use canonical SMILES to uniquely identify the fragment's structure
            fragment_smiles = map(Chem.MolToSmiles, fragments)
            path_fragment_smiles.append((path, set(fragment_smiles)))
    # Select the path avoiding unique fragments
    max_fragments = opts.select_assymetry(map(len, list(zip(*path_fragment_smiles))[1]))
    path_max_fragments = [path for path, frags in path_fragment_smiles if len(frags) == max_fragments]
    if opts.debug:
        debug_info[f'path_fragments_for_asymmetry'] = path_fragment_smiles
        debug_info[f'{str_fn(opts.select_assymetry)}_asymmetry_fragments'] = max_fragments
        debug_info[f'paths_with_{str_fn(opts.select_assymetry)}_asymmetry_fragments'] = path_max_fragments
        debug_info[f'paths_without_{str_fn(opts.select_assymetry)}_asymmetry_fragments'] = [path for path, frags in path_fragment_smiles if len(frags) != max_fragments]
    if len(path_max_fragments) == 1:
        if opts.debug:
            debug_info["result"] = f'unique path with {str_fn(opts.select_assymetry)} fragment asymmetry'
            return path_max_fragments[0], debug_info
        return path_max_fragments[0]
    if opts.debug:
        debug_info["result"] = f'{len(path_max_fragments)} non unique paths found, retruning the first one'
        return path_max_fragments[0], debug_info
    # return max(path_fragment_smiles, key=lambda x: len(x[1]))[0]
    return path_max_fragments[0]


def get_shortest_shortest_path(mol: Chem.Mol, index: int, possible_endpoints: list[int]) -> list[int]:
    """Find the shortest of the shortest paths between an atom and a list of atoms in a molecule.

    :param mol: molecule
    :param index: index of the atom to start the shortest path from
    :param possible_endpoints: list of candidate atom indices as endpoints for the shortest path
    """
    if not isinstance(mol, Chem.Mol):
        raise ValueError("Molecule is a valid RDKit Chem.Mol.")
    if not isinstance(index, int):
        raise ValueError("index is not a valid integer.")
    if not isinstance(possible_endpoints, list):
        raise ValueError("possible_endpoints is not a valid list.")
    result_len = []
    for endpoint in possible_endpoints:
        result_tmp = Chem.GetShortestPath(mol, index, endpoint)
        result_len.append(result_tmp)
    result = min(result_len, key=len)
    return list(result)


def get_longest_shortest_path_from_atom(mol: Chem.Mol, index: int, atoms_to_omit: list[int] = None) -> list[int]:
    """
    Determines the longest shortest path in a molecule, starting from a given
    atom and reaching any other valid atom. This identifies the path to the
    most distant atom from the starting point.

    :param mol: The RDKit molecule to search within.
    :param index: The index of the atom from which all paths should start.
    :param atoms_to_omit: A list of atom indices to exclude from the graph during pathfinding.
    :return: A list of atom indices representing the longest shortest path, including the root atom of the given `index`.
    """
    # Input validation
    if not isinstance(mol, Chem.Mol):
        raise ValueError('A valid RDKit molecule must be provided.')
    if not (0 <= index < mol.GetNumAtoms()):
        raise ValueError(f'Start atom index {index} is out of bounds.')
    if not isinstance(atoms_to_omit, list) or not all(isinstance(atom, int) for atom in atoms_to_omit):
        raise ValueError('atoms_to_omit must be a list of atom indices.')
    if index in atoms_to_omit:
        raise ValueError(f'Start atom index {index} is provided as an atom to omit.')
    # Drop duplicated atoms o omit
    atoms_to_omit_set = set(atoms_to_omit) if atoms_to_omit else set()
    # Create a subgraph excluding omitted atoms and map indices
    map_new_to_old = {}
    map_old_to_new = {}
    subgraph_mol = Chem.RWMol()
    for atom in mol.GetAtoms():
        old_idx = atom.GetIdx()
        if old_idx not in atoms_to_omit_set:
            new_idx = subgraph_mol.AddAtom(atom)
            map_new_to_old[new_idx] = old_idx
            map_old_to_new[old_idx] = new_idx
    for bond in mol.GetBonds():
        begin_idx_old, end_idx_old = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin_idx_old in map_old_to_new and end_idx_old in map_old_to_new:
            begin_idx_new = map_old_to_new[begin_idx_old]
            end_idx_new = map_old_to_new[end_idx_old]
            subgraph_mol.AddBond(begin_idx_new, end_idx_new, bond.GetBondType())
    # If the resulting subgraph is empty or has only one atom, no path can be found.
    if subgraph_mol.GetNumAtoms() <= 1:
        return [index]
    start_atom_new_idx = map_old_to_new[index]
    # Find the longest of the shortest paths in the subgraph
    longest_path_in_subgraph = []
    # Iterate through all atoms in the subgraph as potential endpoints
    for end_atom_new_idx in range(subgraph_mol.GetNumAtoms()):
        if start_atom_new_idx == end_atom_new_idx:
            continue
        # Calculate the single shortest path from the start to this endpoint
        current_shortest_path = Chem.GetShortestPath(subgraph_mol, start_atom_new_idx, end_atom_new_idx)
        # If this path is the longest one so far, it becomes the new candidate
        if len(current_shortest_path) > len(longest_path_in_subgraph):
            longest_path_in_subgraph = list(current_shortest_path)
    # Translate path back to original indices
    if not longest_path_in_subgraph:
        return [index]
    original_indices_path = [index] + [map_new_to_old[idx] for idx in longest_path_in_subgraph]
    return original_indices_path


def extend_path(mol: Chem.Mol, path: list[int], possible_endpoints: list[int]) -> list[int]:
    """Extend a path to include any connected atom that is not part of the proposed endpoints.

    :param mol: molecule
    :param path: the path to extend including the endpoint that is not part of the final solution
    :param possible_endpoints: atom indices that cannot be part of the solution
    :return: the indices of atoms part of the extended path that does not contain any endpoint.
    """
    # Identify the core path (part of the final result)
    path_core_indices = set(path[:-1])
    # Start the traversal from these core atoms
    atoms_to_visit_queue = deque(path_core_indices)
    # Keep track of all traversed atoms to avoid cycles
    # Initialize with all atoms from the path to prevent walking backward
    visited_indices = set(path)
    # Store the newly discovered atoms during the extension
    extended_indices = set()
    # Extend the path via Breadth-First Search (BFS)
    while atoms_to_visit_queue:
        current_idx = atoms_to_visit_queue.popleft()
        current_atom = mol.GetAtomWithIdx(current_idx)
        for neighbor in current_atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            # Ignore if atom already traversed
            if neighbor_idx in visited_indices:
                continue
            # Mark the atom as visited immediately to prevent re-adding to the queue
            visited_indices.add(neighbor_idx)
            # Crucial stopping condition: do not an endpoint
            if neighbor_idx in possible_endpoints:
                continue
            # New atom: add it to the results and the queue for further traversal
            extended_indices.add(neighbor_idx)
            atoms_to_visit_queue.append(neighbor_idx)
    # The final result is the union of the original path core and all the found branches
    return list(path_core_indices.union(extended_indices))
