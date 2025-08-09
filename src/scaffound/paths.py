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


def get_min_max_shortest_path(
        mol: Chem.Mol,
        indices: list[int],
        core: list[int] = None,
        opts: MinMaxShortestPathOptions = None) -> list[int] | tuple[list[int], dict]:
    """
    Find the longest/shortest path between any two points of a list of atom indices,
    with a tie-breaking rule based on a core.

    When multiple paths have the same longest/shortest length, it prefers paths that,
    after removing atoms belonging to the core, result in chemically
    unique fragments.

    :param mol: The molecule to search within.
    :param possible_endpoints: List of candidate atom indices for the endpoint.
    :param core: A list of atom indices representing a scaffold. Used for tie-breaking. Defaults to empty.
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
    debug_info = OrderedDict({'input_smiles': Chem.MolToSmiles(mol, canonical=False)})
    # Candidate path finding
    _, candidate_paths, _, debug_info_ = _find_all_paths_with_minmax_length(mol=mol, indices=indices, core=core, opts=opts)
    if opts.debug:
        debug_info |= debug_info_
    # List of tie-breaker functions to apply in order
    if opts.original_algorithm:
        tie_breakers = [_break_tie_by_asymmetry]
    else:
        tie_breakers = [
            _break_tie_by_ring_count,
            _break_tie_by_ring_size,
            _break_tie_by_aromaticity,
            _break_tie_by_total_ring_atoms,
            _break_tie_by_asymmetry,
            _break_tie_by_total_atomic_number,
            _break_tie_by_atypical_isotopes,
            _break_tie_by_atomic_num_topology,
            _break_tie_by_bond_order_topology,
        ]
    # Tie breaking
    for breaker in tie_breakers:
        if len(candidate_paths) <= 1:
            break
        _, candidate_paths, _, debug_info_ = breaker(mol, candidate_paths, opts)
        if opts.debug:
            debug_info |= debug_info_
    # No unique path found
    if len(candidate_paths) > 1:
        # All paths should be symmetrically identical, choose one
        debug_info['result'] = f'no unique path found; returning the first one'
    # No path found
    elif len(candidate_paths) == 0:
        candidate_paths = [[]]
    # Return
    if opts.debug:
        return (candidate_paths[0] if candidate_paths else []), debug_info
    return candidate_paths[0] if candidate_paths else []


def _find_all_paths_with_minmax_length(
        mol: Chem.Mol,
        indices: list[int],
        core: list[int],
        opts: MinMaxShortestPathOptions) -> tuple[int, list[int], list[int], dict]:
    """Find all longest or shortest shortest paths in a molecule.
    
    :param mol: the molecule
    :param indices: a list of atom indices, pairs of which define the extremities of the path(s) to identify;
    if no path can be identified, then all paths with one extremity starting with one of these indices and with the other
    extremity belonging to the `core` are checked.
    :param core: a list of possible atom indices that the identified path(s) must go through.
    :param opt: options for the selection of the path(s)
    :return: a tuple of the minimum (or maximum) length of the identified path(s), the identified path(s),
    the rejected_path(s), and a dictionary of debugging information.
    """
    debug_info = OrderedDict()
    # Drop atom duplicates from the basic scaffold
    core_set = set(core) or set()
    # Find all paths and their lengths
    all_paths = []
    for start, end in combinations(indices, 2):
        # Ensure start and end are not the same and path goes through the core
        path = Chem.GetShortestPath(mol, start, end)
        if path and not core_set.isdisjoint(path):
            all_paths.append(list(path))
    debug_info["all_paths_terminal_atoms"] = all_paths
    if not all_paths:
        # No path between a pair of extremities exists
        # Retry with one of them belonging to the core
        for start, end in product(indices, core_set):
            if start == end:
                continue
            # Ensure start and end are not the same
            path = Chem.GetShortestPath(mol, start, end)
            if path:
                all_paths.append(list(path))
        if not all_paths:
            # Cannot find a path
            debug_info["result"] = 'no path found'
            return None, [], [], debug_info
        debug_info["all_paths_terminal_and_scaffold_atoms"] = all_paths
    # Identify all paths with the maximum/minimum length
    minmax_len = opts.select_path_len(map(len, all_paths))
    selected_paths = [p for p in all_paths if len(p) == minmax_len]
    rejected_paths = [p for p in all_paths if len(p) != minmax_len]
    debug_info[f'{str_fn(opts.select_path_len)}_path_len'] = minmax_len
    debug_info[f'paths_with_{str_fn(opts.select_path_len)}_len'] = selected_paths
    debug_info[f'paths_without_{str_fn(opts.select_path_len)}_len'] = rejected_paths
    if len(selected_paths) <= 1:
        debug_info["result"] = (f'unique {str_fn(opts.select_path_len)} path found'
                                if len(selected_paths)
                                else 'no path found')
    return minmax_len, selected_paths, rejected_paths, debug_info


def _apply_tie_breaker(
    candidate_paths: list[list[int]],
    metric_fn: callable,
    selection_fn: callable,
    *,  # Makes subsequent arguments keyword-only for clarity
    debug_name: str = None,
    custom_debug_formatter_fn: callable = None
) -> tuple[any, list[int], list[int], dict]:
    """Generic helper to apply any tie-breaking rule.
    
    :param candidate_paths: list of all paths to be evaluated
    :param metric_fn: metric to apply to paths
    :param selection_fn: selection function (min or max) to apply to path metrics
    :param debug_name: name of the tie-breaking rule
    :param custom_debug_formatter_fn: custom debugger function
    """
    # Check arguments
    assert debug_name or custom_debug_formatter_fn, \
        "Must provide 'debug_name' for default debugging, or a 'custom_debug_formatter_fn'."
    # Calculate metrics for each path
    paths_with_metrics = [(path, metric_fn(path)) for path in candidate_paths]
    if not paths_with_metrics:
        return None, [], candidate_paths, OrderedDict({"result": "no path found"})
    # Filter paths
    valid_metrics = [metric for _, metric in paths_with_metrics if metric is not None]
    if not valid_metrics:
        return None, [], candidate_paths, OrderedDict({"result": "no path found"})
    # Select paths
    winning_metric = selection_fn(valid_metrics)
    selected_paths = [path for path, metric in paths_with_metrics if metric == winning_metric]
    rejected_paths = [path for path, metric in paths_with_metrics if metric != winning_metric]
    # Log debug info
    if custom_debug_formatter_fn:
        # Use the provided custom formatter
        debug_info = custom_debug_formatter_fn(winning_metric=winning_metric,
                                               selected=selected_paths,
                                               rejected=rejected_paths,
                                               selection_fn=selection_fn,
                                               paths_with_metrics=paths_with_metrics)
    else:
        # Use the default, built-in formatter
        sel_fn_str = str_fn(selection_fn)
        debug_info = OrderedDict()
        debug_info[f'{sel_fn_str}_{debug_name}'] = winning_metric
        debug_info[f'paths_with_{sel_fn_str}_{debug_name}'] = selected_paths
        debug_info[f'paths_without_{sel_fn_str}_{debug_name}'] = rejected_paths
        if len(selected_paths) <= 1:
            debug_info["result"] = (f'unique path with {sel_fn_str} {debug_name.replace("_", " ")} found'
                                    if len(selected_paths)
                                    else 'no path found')
    return winning_metric, selected_paths, rejected_paths, debug_info


def _break_tie_by_ring_count(mol, candidate_paths, opts):
    """Determine paths, from given candidates, that satisfy the ring count selection criterion.
    
    :param mol: the molecule from which the paths are identified
    :param candidate_paths: candidate paths to investigate
    :param opts: selection criteria.
    :return: a tuple of the minimum (or maximum) ring count in the identified path(s), the identified path(s),
    the rejected_path(s), and a dictionary of debugging information.
    """
    # Map each atom to the rings it belongs to
    ring_info = mol.GetRingInfo()
    atom_to_rings_map = [[] for _ in range(mol.GetNumAtoms())]
    for ring_idx, ring_atoms in enumerate(ring_info.AtomRings()):
        for atom_idx in ring_atoms:
            atom_to_rings_map[atom_idx].append(ring_idx)
    # Define the metric: count the number of unique rings a path passes through
    metric = lambda path: len(set(ring for idx in path for ring in atom_to_rings_map[idx]))
    # Call the generic helper using the default debugger
    return _apply_tie_breaker(candidate_paths=candidate_paths,
                              metric_fn=metric,
                              selection_fn=opts.select_ring_count,
                              debug_name='ring_count')


def _break_tie_by_ring_size(mol, candidate_paths, opts):
    """Determine paths, from given candidates, that satisfy the total ring size selection criterion.
    
    :param mol: the molecule from which the paths are identified
    :param candidate_paths: candidate paths to investigate
    :param opts: selection criteria.
    :return: a tuple of the minimum (or maximum) total ring size in the identified path(s), the identified path(s),
    the rejected_path(s), and a dictionary of debugging information.
    """
    # Map each atom to the rings it belongs to
    ring_info = mol.GetRingInfo()
    atom_to_rings_map = [[] for _ in range(mol.GetNumAtoms())]
    ringsize_map = {}
    for ring_idx, ring_atoms in enumerate(ring_info.AtomRings()):
        for atom_idx in ring_atoms:
            atom_to_rings_map[atom_idx].append(ring_idx)
        ringsize_map[ring_idx] = len(ring_atoms)
    # Define the metric: calculate the sum of sizes of all unique rings the path intersects
    def metric(path):
        visited_rings = {ring for idx in path for ring in atom_to_rings_map[idx]}
        return sum(ringsize_map[ring_idx] for ring_idx in visited_rings)
    # Call the generic helper using the default debugger
    return _apply_tie_breaker(candidate_paths=candidate_paths,
                              metric_fn=metric,
                              selection_fn=opts.select_ring_size,
                              debug_name='total_ring_size')


def _break_tie_by_aromaticity(mol, candidate_paths, opts):
    """Determine paths, from given candidates, that satisfy the aromatic ring count selection criterion.
    
    :param mol: the molecule from which the paths are identified
    :param candidate_paths: candidate paths to investigate
    :param opts: selection criteria.
    :return: a tuple of the minimum (or maximum) ring count in the identified path(s), the identified path(s),
    the rejected_path(s), and a dictionary of debugging information.
    :return: a tuple of the minimum (or maximum) aromatic ring count in the identified path(s), the identified path(s),
    the rejected_path(s), and a dictionary of debugging information.

    """
    # Map each atom to the rings it belongs to
    ring_info = mol.GetRingInfo()
    atom_to_rings_map = [[] for _ in range(mol.GetNumAtoms())]
    ring_aromat_map = {}
    for ring_idx, ring_atoms in enumerate(ring_info.AtomRings()):
        for atom_idx in ring_atoms:
            atom_to_rings_map[atom_idx].append(ring_idx)
        ring_aromat_map[ring_idx] = all(mol.GetAtomWithIdx(atom).GetIsAromatic() for atom in ring_atoms)
    # Define the metric: calculate the number of unique aromatic rings the path intersects
    def metric(path):
        visited_rings = {ring for idx in path for ring in atom_to_rings_map[idx]}
        return sum(ring_aromat_map.get(ring_idx, 0) for ring_idx in visited_rings)
    # Call the generic helper using the default debugger
    return _apply_tie_breaker(candidate_paths=candidate_paths,
                              metric_fn=metric,
                              selection_fn=opts.select_arom_rings,
                              debug_name='aromatic_ring_count')


def _break_tie_by_total_ring_atoms(mol, candidate_paths, opts):
    """Determine paths, from given candidates, that satisfy the total ring atom selection criterion.
    
    :param mol: the molecule from which the paths are identified
    :param candidate_paths: candidate paths to investigate
    :param opts: selection criteria.
    :return: a tuple of the minimum (or maximum) total ring atom count in the identified path(s), the identified path(s),
    the rejected_path(s), and a dictionary of debugging information.
    """
    # Define the metric: calculate the number of ring atoms in the path
    metric = lambda path: sum(1 for idx in path if mol.GetAtomWithIdx(idx).IsInRing())
    # Call the generic helper using the default debugger
    return _apply_tie_breaker(candidate_paths=candidate_paths,
                              metric_fn=metric,
                              selection_fn=opts.select_num_ring_atoms,
                              debug_name='ring_atom_count')


def _break_tie_by_asymmetry(mol, candidate_paths, opts):
    """Determine paths, from given candidates, that satisfy the asymmetry selection criterion.
    
    :param mol: the molecule from which the paths are identified
    :param candidate_paths: candidate paths to investigate
    :param opts: selection criteria.
    :return: a tuple of the number of asymmetrical fragments in the identified path(s), the identified path(s),
    the rejected_path(s), and a dictionary of debugging information.
    """
    # Get the basic scaffold
    true_bsc_atoms = set(scaffolds.get_basic_scaffold(mol, only_atom_indices=True))
    # Canonical SMILES for the fragments of each path
    path_fragment_smiles = []
    for path in candidate_paths:
        # Determine the atoms in the fragment (path atoms minus scaffold atoms)
        fragment_indices = sorted(list(set(path) - true_bsc_atoms))
        if fragment_indices:
            # Create a new molecule from the fragment indices
            rw_mol = Chem.RWMol(mol)
            # Remove all atoms NOT in our fragment
            atoms_to_remove = [i for i in range(mol.GetNumAtoms()) if i not in fragment_indices]
            scaffolds.utils.unassign_chirality_and_delete(rw_mol, atoms_to_remove)
            # Get fragments
            fragments = Chem.GetMolFrags(rw_mol.GetMol(), asMols=True)
            # Use canonical SMILES to uniquely identify the fragment's structure
            fragment_smiles_set = set(map(Chem.MolToSmiles, fragments))
            path_fragment_smiles.append((path, fragment_smiles_set))
    # Create a lookup map for the metric function
    path_to_frags_map = {tuple(path): frags for path, frags in path_fragment_smiles}
    # Define the metric: calculate the number of unique fragments
    metric = lambda path: len(path_to_frags_map.get(tuple(path), set()))
    #Define the custom debug formatter
    def _debug_formatter(winning_metric, selected, rejected, selection_fn, **kwargs) -> OrderedDict:
        debug_info = OrderedDict()
        sel_fn_str = str_fn(selection_fn)
        # Add the custom debug information that the default debugger can't handle
        debug_info['path_fragments_for_asymmetry'] = path_fragment_smiles
        # Add the standard debug information
        debug_info[f'{sel_fn_str}_asymmetry_fragments'] = winning_metric
        debug_info[f'paths_with_{sel_fn_str}_asymmetry_fragments'] = selected
        debug_info[f'paths_without_{sel_fn_str}_asymmetry_fragments'] = rejected
        if len(selected) <= 1:
            debug_info["result"] = (f'unique path with {sel_fn_str} fragment asymmetry found'
                                    if len(selected)
                                    else 'no path found')
        return debug_info
    # Call the generic helper with the custom formatter ---
    return _apply_tie_breaker(candidate_paths=candidate_paths,
                              metric_fn=metric,
                              selection_fn=opts.select_assymetry,
                              # Use the custom formatter
                              custom_debug_formatter_fn=_debug_formatter)


def _break_tie_by_total_atomic_number(mol, candidate_paths, opts):
    """Determine paths, from given candidates, that satisfy the total atomic number selection criterion.
    
    :param mol: the molecule from which the paths are identified
    :param candidate_paths: candidate paths to investigate
    :param opts: selection criteria.
    :return: a tuple of the minimum (or maximum) total atomic number in the identified path(s), the identified path(s),
    the rejected_path(s), and a dictionary of debugging information.
    """
    # Define the metric: calculate the sum of atomic numbers along the path
    metric = lambda path: sum(mol.GetAtomWithIdx(idx).GetAtomicNum() for idx in path)
    # Call the generic helper using the default debugger
    return _apply_tie_breaker(candidate_paths=candidate_paths,
                              metric_fn=metric,
                              selection_fn=opts.select_total_atomic_num,
                              debug_name='total_atomic_number')


def _break_tie_by_atypical_isotopes(mol, candidate_paths, opts):
    """Determine paths, from given candidates, that satisfy the total atypical isotope selection criterion.
    
    :param mol: the molecule from which the paths are identified
    :param candidate_paths: candidate paths to investigate
    :param opts: selection criteria.
    :return: a tuple of the minimum (or maximum) total atypical isotope in the identified path(s), the identified path(s),
    the rejected_path(s), and a dictionary of debugging information.
    """
    # Define the metric: calculate the sum of atypical isotopes along the path
    periodic_table = Chem.GetPeriodicTable()
    metric = lambda path: sum(mol.GetAtomWithIdx(idx).GetIsotope()
                              for idx in path
                              if periodic_table.GetMostCommonIsotope(mol.GetAtomWithIdx(idx).GetAtomicNum()) != mol.GetAtomWithIdx(idx).GetIsotope())
    # Call the generic helper using the default debugger
    return _apply_tie_breaker(candidate_paths=candidate_paths,
                              metric_fn=metric,
                              selection_fn=opts.select_isotopes,
                              debug_name='total_atypical_isotopic_mass')


def _break_tie_by_atomic_num_topology(mol, candidate_paths, opts):
    """Determine paths, from given candidates, that satisfy the atomic number topology selection criterion.
    
    :param mol: the molecule from which the paths are identified
    :param candidate_paths: candidate paths to investigate
    :param opts: selection criteria.
    :return: a tuple of the minimum (or maximum) atomic number topology in the identified path(s), the identified path(s),
    the rejected_path(s), and a dictionary of debugging information.
    """
    # Define the metric: get the sequence of atomic numbers, then choose the forward
    # or reverse sequence based on the options. The sequence itself is the metric.
    def metric(path):
        seq_atomnum = [mol.GetAtomWithIdx(idx).GetAtomicNum() for idx in path]
        return opts.select_atom_num_topology_dir(seq_atomnum, seq_atomnum[::-1])
    # Call the generic helper using the default debugger
    return _apply_tie_breaker(candidate_paths=candidate_paths,
                             metric_fn=metric,
                             selection_fn=opts.select_atom_num_topology,
                             debug_name='atomic_number_topology')


def _break_tie_by_bond_order_topology(mol, candidate_paths, opts):
    """Determine paths, from given candidates, that satisfy the ring count selection criterion.
    
    :param mol: the molecule from which the paths are identified
    :param candidate_paths: candidate paths to investigate
    :param opts: selection criteria.
    :return: a tuple of the minimum (or maximum) ring count in the identified path(s), the identified path(s),
    the rejected_path(s), and a dictionary of debugging information.
    :return: a tuple of the minimum (or maximum) ring count in the identified path(s), the identified path(s),
    the rejected_path(s), and a dictionary of debugging information.
    """
    # Define the metric: get the sequence of bond orders, then choose the forward
    # or reverse sequence based on the options. The sequence itself is the metric.
    def metric(path):
        seq_bondorder = [mol.GetBondBetweenAtoms(path[i - 1], path[i]).GetBondTypeAsDouble()
                         for i in range(1, len(path))]
        return opts.select_bond_order_topology_dir(seq_bondorder, seq_bondorder[::-1])
    # Call the generic helper using the default debugger
    return _apply_tie_breaker(candidate_paths=candidate_paths,
                             metric_fn=metric,
                             selection_fn=opts.select_bond_order_topology,
                             debug_name='bond_order_topology')


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
