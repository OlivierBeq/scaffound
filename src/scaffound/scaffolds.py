# -*- coding: utf-8 -*-


"""Implementation of the scaffold definitions of Dompé's 'Molecular Anatomy'."""

from collections import deque
from functools import cmp_to_key
from itertools import chain

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.rdBase import BlockLogs

from .cip import compare_substituents_bfs
from . import paths


class MolecularAnatomy:
    """Class implementing the scaffold definitions of Dompé's 'Molecular Anatomy'.

    Reimplemented from:

        Manelfi, C., Gemei, M., Talarico, C. et al.
        “Molecular Anatomy”: a new multi-dimensional hierarchical scaffold analysis tool.
        J Cheminform 13, 54 (2021).
        https://doi.org/10.1186/s13321-021-00526-y
    """

    def __init__(self, mol: Chem.Mol, loose_ez_stereo: bool = False, opts: paths.MinMaxShortestPathOptions = None):
        """Create the anatomy of a molecule.

        :param mol: Molecule from which to obtain the scaffold anatomy.
        :param loose_ez_stereo: Omit cis/trans stereo in scaffolds (as in the `Molecular Anatomy`).
        """
        self.mol = mol
        self.opts = opts or paths.MinMaxShortestPathOptions()
        # Get the basic scaffold
        self._bs = get_basic_scaffold(mol, loose_ez_stereo=loose_ez_stereo)
        # bs_ids = get_basic_scaffold(mol, loose_ez_stereo=loose_ez_stereo, only_atom_indices=True)
        # Get the decorated scaffold
        self._ds = get_decorated_scaffold(mol, loose_ez_stereo=loose_ez_stereo)
        # ds_ids = get_decorated_scaffold(mol, basic_scaffold_atoms=bs_ids, loose_ez_stereo=loose_ez_stereo, only_atom_indices=True)
        # Get the augmented scaffolds
        self._as = get_augmented_scaffold(mol, loose_ez_stereo=loose_ez_stereo, opts=opts)
        # Obtain the frameworks
        self._bf = get_generic_graph(self._bs)
        self._df = get_generic_graph(self._ds)
        self._af = get_generic_graph(self._as)
        # Obtain the wireframes
        self._bw = get_generic_graph(get_saturated_graph(self._bs))
        self._dw = get_generic_graph(get_saturated_graph(self._ds))
        self._aw = get_generic_graph(get_saturated_graph(self._as))

    @property
    def basic_scaffold(self):
        return self._bs

    @property
    def decorated_scaffold(self):
        return self._ds

    @property
    def augmented_scaffold(self):
        return self._as

    @property
    def basic_framework(self):
        return self._bf

    @property
    def decorated_framework(self):
        return self._df

    @property
    def augmented_framework(self):
        return self._af

    @property
    def basic_wireframe(self):
        return self._bw

    @property
    def decorated_wireframe(self):
        return self._dw

    @property
    def augmented_wireframe(self):
        return self._aw

    @property
    def generic_graph(self):
        return get_generic_graph(self.mol)

    @property
    def saturated_graph(self):
        return get_saturated_graph(self.mol)

    @property
    def wireframe_graph(self):
        return get_generic_graph(get_saturated_graph(self.mol))

    def to_dict(self, original: bool = False):
        """Return the Molecular Anatomy as a dictionary.

        :param original: If `True`, return only the basic, decorated and augmented scaffolds, frameworks and wireframes.
        Otherwise, include the saturated and generic graphs of the molecule and their scaffolds, frameworks and wireframes.
        """
        if not original:
            saturated_graph = get_saturated_graph(self.mol)
            generic_graph = get_generic_graph(self.mol)
            saturated_anatomy = MolecularAnatomy(saturated_graph, opts=self.opts)
            generic_anatomy = MolecularAnatomy(generic_graph, opts=self.opts)
            return {'basic scaffold': self.basic_scaffold,
                    'decorated scaffold': self.decorated_scaffold,
                    'augmented scaffold': self.augmented_scaffold,
                    'basic_framework': self.basic_framework,
                    'decorated_framework': self.decorated_framework,
                    'augmented_framework': self.augmented_framework,
                    'basic wireframe': self.basic_wireframe,
                    'decorated wireframe': self.decorated_wireframe,
                    'augmented wireframe': self.augmented_wireframe,
                    'saturated graph': saturated_graph,
                    'saturated basic scaffold': saturated_anatomy.basic_scaffold,
                    'saturated augmented scaffold': saturated_anatomy.augmented_scaffold,
                    'saturated basic framework': saturated_anatomy.basic_framework,
                    'saturated augmented framework': saturated_anatomy.augmented_framework,
                    'generic graph': generic_graph,
                    'generic augmented scaffold': generic_anatomy.augmented_scaffold,
                    'generic augmented wireframe': generic_anatomy.augmented_wireframe,
                    }
        return {'basic scaffold': self.basic_scaffold,
                'decorated scaffold': self.decorated_scaffold,
                'augmented scaffold': self.augmented_scaffold,
                'basic_framework': self.basic_framework,
                'decorated_framework': self.decorated_framework,
                'augmented_framework': self.augmented_framework,
                'basic wireframe': self.basic_wireframe,
                'decorated wireframe': self.decorated_wireframe,
                'augmented wireframe': self.augmented_wireframe,
                }

    def to_pandas(self, original: bool = False):
        """Return the Molecular Anatomy as a Pandas Series.

        :param original: If `True`, return only the basic, decorated and augmented scaffolds, frameworks and wireframes.
        Otherwise, include the saturated and generic graphs of the molecule and their scaffolds, frameworks and wireframes.
        """
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ImportError('Missing optional dependency \'pandas\'.  Use pip or conda to install pandas.')
        return pd.Series(self.to_dict(original=original))

    def as_table(self, original: bool = False):
        """Format the Molecular Anatomy as Table S1 provided by authors of the seminal article.

        :param original: If `True`, return only the basic, decorated and augmented scaffolds, frameworks and wireframes.
        Otherwise, include the saturated and generic graphs of the molecule and their scaffolds, frameworks and wireframes.
        """
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ImportError('Missing optional dependency \'pandas\'.  Use pip or conda to install pandas.')
        if not original:
            saturated_graph = get_saturated_graph(self.mol)
            generic_graph = get_generic_graph(self.mol)
            saturated_anatomy = MolecularAnatomy(saturated_graph, opts=self.opts)
            generic_anatomy = MolecularAnatomy(generic_graph, opts=self.opts)
            return pd.Series({'Molecule_SMILES': Chem.MolToSmiles(self.mol),
                              'Molecule_inchikey': Chem.MolToInchiKey(self.mol),
                              'Augmented_Scaffold_inchikey': Chem.MolToInchiKey(self.augmented_scaffold),
                              'Augmented_Scaffold_SMILES': Chem.MolToSmiles(self.augmented_scaffold),
                              'Augmented_Framework_inchikey': Chem.MolToInchiKey(self.augmented_framework),
                              'Augmented_Framework_smiles': Chem.MolToSmiles(self.augmented_framework),
                              'Augmented_Wireframe_inchikey': Chem.MolToInchiKey(self.augmented_wireframe),
                              'Augmented_Wireframe_smiles': Chem.MolToSmiles(self.augmented_wireframe),
                              'Decorated_Scaffold_inchikey': Chem.MolToInchiKey(self.decorated_scaffold),
                              'Decorated_Scaffold_SMILES': Chem.MolToSmiles(self.decorated_scaffold),
                              'Decorated_Framework_inchikey': Chem.MolToInchiKey(self.decorated_framework),
                              'Decorated_Framework_SMILES': Chem.MolToSmiles(self.decorated_framework),
                              'Decorated_Wireframe_inchikey': Chem.MolToInchiKey(self.decorated_wireframe),
                              'Decorated_Wireframe_SMILES': Chem.MolToSmiles(self.decorated_wireframe),
                              'Basic_Scaffold_inchikey': Chem.MolToInchiKey(self.basic_scaffold),
                              'Basic_Scaffold_SMILES': Chem.MolToSmiles(self.basic_scaffold),
                              'Basic_Framework_inchikey': Chem.MolToInchiKey(self.basic_framework),
                              'Basic_Framework_SMILES': Chem.MolToSmiles(self.basic_framework),
                              'Basic_Wireframe_inchikey': Chem.MolToInchiKey(self.basic_wireframe),
                              'Basic_Wireframe_SMILES': Chem.MolToSmiles(self.basic_wireframe),
                              'Saturated_graph_inchikey': Chem.MolToInchiKey(saturated_graph),
                              'Saturated_graph_SMILES': Chem.MolToSmiles(saturated_graph),
                              'Saturated_graph_Augmented_Scaffold_inchikey': Chem.MolToInchiKey(saturated_anatomy.augmented_scaffold),
                              'Saturated_graph_Augmented_Scaffold_SMILES': Chem.MolToSmiles(saturated_anatomy.augmented_scaffold),
                              'Saturated_graph_Augmented_Framework_inchikey': Chem.MolToInchiKey(saturated_anatomy.augmented_framework),
                              'Saturated_graph_Augmented_Framework_smiles': Chem.MolToSmiles(saturated_anatomy.augmented_framework),
                              'Saturated_graph_Basic_Scaffold_inchikey': Chem.MolToInchiKey(saturated_anatomy.basic_scaffold),
                              'Saturated_graph_Basic_Scaffold_SMILES': Chem.MolToSmiles(saturated_anatomy.basic_scaffold),
                              'Saturated_graph_Basic_Framework_inchikey': Chem.MolToInchiKey(saturated_anatomy.basic_framework),
                              'Saturated_graph_Basic_Framework_SMILES': Chem.MolToSmiles(saturated_anatomy.basic_framework),
                              'Generic_graph_inchikey': Chem.MolToInchiKey(generic_graph),
                              'Generic_graph_SMILES': Chem.MolToSmiles(generic_graph),
                              'Generic_graph_Augmented_Scaffold_inchikey': Chem.MolToInchiKey(generic_anatomy.augmented_scaffold),
                              'Generic_graph_Augmented_Scaffold_SMILES': Chem.MolToSmiles(generic_anatomy.augmented_scaffold),
                              'Generic_graph_Augmented_Wireframe_inchikey': Chem.MolToInchiKey(generic_anatomy.augmented_wireframe),
                              'Generic_graph_Augmented_Wireframe_smiles': Chem.MolToSmiles(generic_anatomy.augmented_wireframe),
                              })
        return pd.Series({'Molecule_SMILES': Chem.MolToSmiles(self.mol),
                          'Molecule_inchikey': Chem.MolToInchiKey(self.mol),
                          'Augmented_Scaffold_inchikey': Chem.MolToInchiKey(self.augmented_scaffold),
                          'Augmented_Scaffold_SMILES': Chem.MolToSmiles(self.augmented_scaffold),
                          'Augmented_Framework_inchikey': Chem.MolToInchiKey(self.augmented_framework),
                          'Augmented_Framework_smiles': Chem.MolToSmiles(self.augmented_framework),
                          'Augmented_Wireframe_inchikey': Chem.MolToInchiKey(self.augmented_wireframe),
                          'Augmented_Wireframe_smiles': Chem.MolToSmiles(self.augmented_wireframe),
                          'Decorated_Scaffold_inchikey': Chem.MolToInchiKey(self.decorated_scaffold),
                          'Decorated_Scaffold_SMILES': Chem.MolToSmiles(self.decorated_scaffold),
                          'Decorated_Framework_inchikey': Chem.MolToInchiKey(self.decorated_framework),
                          'Decorated_Framework_SMILES': Chem.MolToSmiles(self.decorated_framework),
                          'Decorated_Wireframe_inchikey': Chem.MolToInchiKey(self.decorated_wireframe),
                          'Decorated_Wireframe_SMILES': Chem.MolToSmiles(self.decorated_wireframe),
                          'Basic_Scaffold_inchikey': Chem.MolToInchiKey(self.basic_scaffold),
                          'Basic_Scaffold_SMILES': Chem.MolToSmiles(self.basic_scaffold),
                          'Basic_Framework_inchikey': Chem.MolToInchiKey(self.basic_framework),
                          'Basic_Framework_SMILES': Chem.MolToSmiles(self.basic_framework),
                          'Basic_Wireframe_inchikey': Chem.MolToInchiKey(self.basic_wireframe),
                          'Basic_Wireframe_SMILES': Chem.MolToSmiles(self.basic_wireframe)
                          })


def get_basic_scaffold(mol: Chem.Mol,
                       loose_ez_stereo: bool = False,
                       only_atom_indices: bool = False) -> Chem.Mol | list[int]:
    """Obtain the basic scaffold by iterative pruning of terminal atoms.

    :param loose_ez_stereo: Omit cis/trans stereo in scaffolds (as in the `Molecular Anatomy`).
    :param only_atom_indices: If true, return only the atoms indices.
    """
    if mol is None:
        raise ValueError("Molecule is None.")
    # Create a copy of the molecule to prune.
    rw_mol = Chem.RWMol(mol)
    AllChem.Kekulize(rw_mol)
    # Remember original atom indices
    if only_atom_indices:
        for atom in rw_mol.GetAtoms():
            atom.SetIntProp('original_atomid', atom.GetIdx())
    periodic_table = Chem.GetPeriodicTable()
    while True:
        # Identify atoms in rings
        ring_atoms_original = {atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.IsInRing()}
        # If there are no rings at all, there is no scaffold.
        if not ring_atoms_original:
            return [] if only_atom_indices else Chem.Mol()
        atoms_removed = []
        # Find all atoms that are eligible for pruning in the current molecule state.
        atom_ids = sorted([atom.GetIdx() for atom in rw_mol.GetAtoms()], reverse=True)
        for atom_id in atom_ids:
            atom: Chem.Atom = rw_mol.GetAtomWithIdx(atom_id)
            # An atom can be pruned if it is terminal (degree 1)...
            if atom.GetDegree() == 1:
                # ...and was not part of a ring.
                if atom.GetIdx() not in ring_atoms_original:
                    atoms_removed.append(atom_id)
                    unassign_chirality_and_delete(rw_mol, [atom_id])
        # If there are no atoms to prune in a full pass, we're done.
        if len(atoms_removed) == 0:
            break
        for atom in rw_mol.GetAtoms():
            atom.SetNoImplicit(False)
    # Exit now if only the atom indices are required
    if only_atom_indices:
        return [atom.GetIntProp('original_atomid') for atom in rw_mol.GetAtoms()]
    # Sanitize the molecule.
    for atom in rw_mol.GetAtoms():
        default_valence = periodic_table.GetDefaultValence(atom.GetAtomicNum())
        if atom.GetTotalValence() < default_valence:
            atom.SetNumExplicitHs(default_valence - atom.GetTotalValence() - atom.GetFormalCharge())
    mol = rw_mol.GetMol()
    with BlockLogs():
        Chem.SanitizeMol(mol, catchErrors=False)
    # Remove cis-trans stereo
    if loose_ez_stereo:
            mol = reconstruct_and_flatten_db_stereo(mol)
    return mol


def get_decorated_scaffold(mol: Chem.Mol,
                           basic_scaffold_atoms: None | list[int] = None,
                           loose_ez_stereo: bool = False,
                           only_atom_indices: bool = False) -> Chem.Mol | list[int]:
    """Obtain the decorated scaffold by iteratively pruning terminal atoms
    that have a bond order of one.

    :param mol: molecule
    :param loose_ez_stereo: Omit cis/trans stereo in scaffolds (as in the `Molecular Anatomy`).
    :param only_atom_indices: If true, return only the atoms indices.
    """
    if mol is None:
        raise ValueError("Molecule is None.")
    # Create a copy of the molecule to prune.
    rw_mol = Chem.RWMol(mol)
    AllChem.Kekulize(rw_mol)
    # Remember original atom indices
    if only_atom_indices:
        for atom in rw_mol.GetAtoms():
            atom.SetIntProp('original_atomid', atom.GetIdx())
    # Flag terminal atoms in the original molecule
    for atom in rw_mol.GetAtoms():
        atom.SetBoolProp('terminalAtom', atom.GetDegree() == 1)
    # Flag atoms part of the basic scaffold
    bsc_atoms = get_basic_scaffold(mol, loose_ez_stereo, True) if basic_scaffold_atoms is None else basic_scaffold_atoms
    # Get fragments not part of the basic scaffold
    rw_mol_no_bsc = Chem.RWMol(rw_mol)
    for atom in rw_mol_no_bsc.GetAtoms():
        atom.SetIntProp('originalAtomId', atom.GetIdx())
    unassign_chirality_and_delete(rw_mol_no_bsc, bsc_atoms)
    with BlockLogs():
        Chem.SanitizeMol(rw_mol_no_bsc)
    frags = AllChem.GetMolFrags(rw_mol_no_bsc, asMols=True)
    # Identify terminal atoms with single bonds from fragments in the original molecule
    frag_atoms_ro_remove = []
    for frag in frags:
        delete_frag = False
        frag_atom_ids = [atom.GetIntProp('originalAtomId') for atom in frag.GetAtoms()]
        for atom_id in frag_atom_ids:
            atom = rw_mol.GetAtomWithIdx(atom_id)
            bonds = atom.GetBonds()
            if len(frag_atom_ids) != 1 or (atom.GetBoolProp('terminalAtom') and bonds[0].GetBondType() == Chem.BondType.SINGLE):
                delete_frag = True
                break
        if delete_frag:
            for atom_id in frag_atom_ids:
                frag_atoms_ro_remove.append(atom_id)
    # Remove atoms of fragments with terminal atoms with single bonds
    unassign_chirality_and_delete(rw_mol, frag_atoms_ro_remove)
    # Exit now if only the atom indices are required
    if only_atom_indices:
        return [atom.GetIntProp('original_atomid') for atom in rw_mol.GetAtoms()]
    # Sanitize the molecule.
    for atom in rw_mol.GetAtoms():
        atom.SetIsAromatic(False)      # Erase any ghost aromatic flags
        atom.SetNoImplicit(False)      # Grant permission to add hydrogens
    mol = rw_mol.GetMol()
    with BlockLogs():
        Chem.SanitizeMol(mol, catchErrors=False)
    # Remove cis-trans stereo
    if loose_ez_stereo:
        mol = reconstruct_and_flatten_db_stereo(mol)
    return mol


def get_augmented_scaffold(mol: Chem.Mol,
                           basic_scaffold_atoms: None | list[int] = None,
                           decorated_scaffold_atoms: None | list[int] = None,
                           loose_ez_stereo: bool = False,
                           opts: paths.MinMaxShortestPathOptions = None) -> Chem.Mol:
    opts = opts or paths.MinMaxShortestPathOptions()
    Chem.Kekulize(mol)
    atoms_to_remove = identify_terminal_atoms(mol)
    rw_mol = Chem.RWMol(mol)
    unassign_chirality_and_delete(rw_mol, atoms_to_remove)
    mol = rw_mol.GetMol()
    with BlockLogs():
        Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol)
    with BlockLogs():
        Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol)
    # Get atoms of the basic scaffold
    bsc_atoms = (get_basic_scaffold(mol, loose_ez_stereo, True)
                 if basic_scaffold_atoms is None
                 else basic_scaffold_atoms)
    # Get atoms of the decorated scaffold
    dsc_atoms = (get_decorated_scaffold(mol, bsc_atoms, loose_ez_stereo, True)
                 if decorated_scaffold_atoms is None
                 else decorated_scaffold_atoms)
    # Fast exit if the molecule is its own basic or decorated scaffold
    if len(bsc_atoms) == len(mol.GetAtoms()):
        mol = fix_valence(mol)
        return mol
    if len(dsc_atoms) == len(mol.GetAtoms()):
        rw_mol = Chem.RWMol(mol)
        unassign_chirality_and_delete(rw_mol, [atom.GetIdx()
                                               for atom in rw_mol.GetAtoms()
                                               if atom.GetIdx() not in dsc_atoms])
        mol = rw_mol.GetMol()
        mol = fix_valence(mol)
        return mol
    # Find true terminal carbon atoms
    true_terminal_carbons = list(chain.from_iterable(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6&D1]'))))
    # Find bespoke terminal carbon atoms
    bespoke_terminal_carbons = [atom.GetIdx() for atom in mol.GetAtoms()
                                if atom.GetIdx() not in dsc_atoms and atom.GetSymbol() == 'C']
    # Only consider bespoke terminal carbons if no true terminal carbons are found
    if len(true_terminal_carbons) > 0:
        # Include bespoke terminal carbons only if they are not part of the longest shortest path
        longest_shortest_path = paths.get_min_max_shortest_path(mol, true_terminal_carbons, bsc_atoms,
                                                                                 opts=opts)
        if opts.debug:
            longest_shortest_path = longest_shortest_path[0]
        terminal_carbons = list(set(true_terminal_carbons) | set(atom
                                                                 for atom in bespoke_terminal_carbons
                                                                 if atom not in longest_shortest_path))
    elif len(bespoke_terminal_carbons) == 1:
        terminal_carbons = bespoke_terminal_carbons
    else:
        # Consider bespoke terminal carbons only if they are not part of the longest shortest path
        longest_shortest_path = paths.get_min_max_shortest_path(mol, bespoke_terminal_carbons, bsc_atoms, opts=opts)
        if opts.debug:
            longest_shortest_path = longest_shortest_path[0]
        terminal_carbons = [atom
                               for atom in bespoke_terminal_carbons
                               if atom not in longest_shortest_path or atom in [longest_shortest_path[0], longest_shortest_path[-1]]]
    # Find the longest shortest path between terminal carbons
    longest_shortest_path = paths.get_min_max_shortest_path(mol, terminal_carbons, bsc_atoms, opts=opts)
    if opts.debug:
        longest_shortest_path = longest_shortest_path[0]
    # Consider the case when the longest path has a length of 3 and contains both terminal carbons
    if len(terminal_carbons) == 2 and len(set(terminal_carbons).difference(longest_shortest_path)) == 0 and len(longest_shortest_path) == 3:
        common_atoms = [x for x in longest_shortest_path if x in terminal_carbons]
        if mol.GetAtomWithIdx(common_atoms[0]).GetSymbol() == 'C':
            new_atom_list_to_remove = terminal_carbons[1:]
            rw_mol = Chem.RWMol(mol)
            unassign_chirality_and_delete(rw_mol, new_atom_list_to_remove)
            mol = rw_mol.GetMol()
            mol = fix_valence(mol)
    else:
        atoms_to_remove = set()
        atoms_to_replace = set()
        # Obtain the union with the longest path
        extended_scaffold = list(set(bsc_atoms).union(longest_shortest_path))
        # Remove all paths that start from the terminal atom and are disjoint of the longest path
        for atom in mol.GetAtoms():
            if len(atom.GetNeighbors()) > 1:
                continue
            atom_id = atom.GetIdx()
            # Removing any side chain of no interest
            if atom_id not in extended_scaffold and atom_id not in terminal_carbons and atom_id not in dsc_atoms:
                path = paths.get_shortest_shortest_path(mol, atom_id, list(set(bsc_atoms + dsc_atoms))) # terminal_carbons
                if set(path[:-1]).isdisjoint(longest_shortest_path):
                    root_atom = mol.GetAtomWithIdx(path[-1])
                    if root_atom.IsInRing() and root_atom.GetSymbol() != 'C' and root_atom.GetIsAromatic():
                        atoms_to_replace.add((path[-1], path[-2]))
                    else:
                        atoms_to_remove |= set(path[:-1])
                # Removing side chains connected to the longest path
                if terminal_carbons:
                    path = paths.get_shortest_shortest_path(mol, atom_id, terminal_carbons)
                    if set(path[:-1]).isdisjoint(longest_shortest_path):
                        root_atom = mol.GetAtomWithIdx(path[-1])
                        x, y, z = root_atom.IsInRing(), root_atom.GetSymbol(), root_atom.GetIsAromatic()
                        if root_atom.IsInRing() and root_atom.GetSymbol() != 'C' and root_atom.GetIsAromatic():
                            atoms_to_replace.add((path[-1], path[-2]))
                        else:
                            atoms_to_remove |= set(path[:-1])
        # Remove any side chain neither part of the scaffold nor the longest path
        for terminal_carbon in terminal_carbons:
            if terminal_carbon not in extended_scaffold:
                # Get the shortest path from the terminal atom and the closets atom of the extended scaffold
                path = paths.get_shortest_shortest_path(mol, terminal_carbon, extended_scaffold)
                # Extend the path to include its side chains but no other terminal carbon
                extended_path = paths.extend_path(mol, path, list(set(extended_scaffold) | set(carbon
                                                                                         for carbon in terminal_carbons
                                                                                         if carbon != terminal_carbon)))
                root_atom = mol.GetAtomWithIdx(path[-1])
                if root_atom.IsInRing() and root_atom.GetSymbol() != 'C' and root_atom.GetIsAromatic():
                    atoms_to_remove |= set(atom for atom in extended_path
                                           if atom != path[-2])
                    # Store the root to add an atom to and the atom to remove
                    if path[-2] not in atoms_to_remove:
                        atoms_to_replace.add((path[-1], path[-2]))
                else:
                    atoms_to_remove |= set(extended_path)
        # Unmark atoms of the decorated scaffold for removal or replacement
        for atom_id in dsc_atoms:
            if atom_id in atoms_to_remove:
                atoms_to_remove.remove(atom_id)
        atoms_to_replace = list(atoms_to_replace)
        for i in range(len(atoms_to_replace)):
            if atoms_to_replace[i][1] in dsc_atoms:
                del atoms_to_replace[i]
        # Keep track of original atom ids after deletion of others
        for atom in mol.GetAtoms():
            atom.SetIntProp('__original_id__', atom.GetIdx())
        # Remove atoms marked from removal
        rw_mol = Chem.RWMol(mol)
        unassign_chirality_and_delete(rw_mol, atoms_to_remove)
        old_to_new_ids = {atom.GetIntProp('__original_id__'): atom.GetIdx()
                          for atom in rw_mol.GetAtoms()}
        for root, atom in atoms_to_replace:
            rw_mol.RemoveAtom(old_to_new_ids[atom])
            w = rw_mol.AddAtom(Chem.Atom('H'))
            rw_mol.AddBond(old_to_new_ids[root], w, Chem.BondType.SINGLE)
            rw_mol.GetAtomWithIdx(old_to_new_ids[root]).UpdatePropertyCache()
        mol = rw_mol.GetMol()
        with BlockLogs():
            Chem.SanitizeMol(mol)
        mol = fix_valence(mol)
    atoms_to_remove = identify_terminal_atoms(mol)
    rw_mol = Chem.RWMol(mol)
    unassign_chirality_and_delete(rw_mol, atoms_to_remove)
    mol = rw_mol.GetMol()
    with BlockLogs():
        Chem.SanitizeMol(mol)
        Chem.SanitizeMol(mol)
    mol = fix_valence(mol)
    return Chem.RemoveHs(mol)


def get_basic_framework(mol: Chem.Mol) -> Chem.Mol:
    return get_generic_graph(get_basic_scaffold(mol))


def get_basic_wireframe(mol: Chem.Mol) -> Chem.Mol:
    return get_generic_graph(get_saturated_graph(get_basic_scaffold(mol)))


def get_decorated_framework(mol: Chem.Mol) -> Chem.Mol:
    return get_generic_graph(get_decorated_scaffold(mol))


def get_decorated_wireframe(mol: Chem.Mol) -> Chem.Mol:
    return get_generic_graph(get_saturated_graph(get_decorated_scaffold(mol)))


def get_augmented_framework(mol: Chem.Mol, opts: paths.MinMaxShortestPathOptions = None) -> Chem.Mol:
    return get_generic_graph(get_augmented_scaffold(mol, opts=opts))


def get_augmented_wireframe(mol: Chem.Mol, opts: paths.MinMaxShortestPathOptions = None) -> Chem.Mol:
    return get_generic_graph(get_saturated_graph(get_augmented_scaffold(mol, opts=opts)))


def get_saturated_graph(mol: Chem.Mol) -> Chem.Mol:
    """Obtain the saturated graph of a molecule by replacing bonds by single bonds
    and dropping formal charges."""
    if mol is None:
        raise ValueError("Molecule is None.")
    # Create a copy of the molecule to prune.
    mol = Chem.Mol(mol)
    # Drop aromaticity
    for atom in mol.GetAtoms():
        atom.SetIsAromatic(False)
        if atom.GetFormalCharge() != 0:
            atom.SetNoImplicit(False)
            atom.SetFormalCharge(0)
    for bond in mol.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    with BlockLogs():
        Chem.SanitizeMol(mol)
    return mol


def get_generic_graph(mol: Chem.Mol) -> Chem.Mol:
    """Obtain the generic graph of a molecule by relacing atoms by carbons."""
    if mol is None:
        raise ValueError("Molecule is None.")
    # First remove sulfonyl groups
    for reaction in ['[#16D4:1](=[O])(=[O])>>[*:1]', '[#15D4:1](=O)>>[*:1]',
                     '[#6:1][#16D6](*)(*)(*)(*)(*)>>[*:1]', '[*D5:1](=O)>>[*:1]']:
        rxn = AllChem.ReactionFromSmarts(reaction)
        rxn.Initialize()
        while rxn.IsMoleculeReactant(mol):
            product = rxn.RunReactants((mol,))
            mol = Chem.Mol(product[0][0]) if len(product) > 0 else mol
            with BlockLogs():
                Chem.SanitizeMol(mol)
    # Remove atoms that are more than tetravalent
    mol = prune_hypervalent_atoms(mol)
    Chem.Kekulize(mol)
    rw_mol = Chem.RWMol(mol)
    for atom in rw_mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            atom.SetAtomicNum(6)
        atom.SetIsotope(0)
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    mol = Chem.RemoveHs(rw_mol.GetMol())
    with BlockLogs():
        Chem.SanitizeMol(mol)
    return mol


def reconstruct_and_flatten_db_stereo(mol):
    # 1. Store the chiral information from the original molecule
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    # chiral_centers is a list of tuples: [(atom_idx, 'R'/'S'), ...]
    # 2. Flatten the molecule into a non-isomeric SMILES string. This
    #    destroys ALL stereochemistry (bond and atom) and ring geometry.
    non_iso_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
    # 3. Rebuild the molecule from the "dumb" SMILES string.
    #    The new molecule has no stereochemistry at all.
    new_mol = Chem.MolFromSmiles(non_iso_smiles)
    # If the rebuild fails, return None
    if new_mol is None:
        return None
    # 4. Re-apply the original chiral information to the new molecule.
    #    We use a dictionary to map atom index to the R/S tag for easy lookup.
    idx_to_chirality = {center[0]: center[1] for center in chiral_centers}
    for atom in new_mol.GetAtoms():
        if atom.GetIdx() in idx_to_chirality:
            # Get the original chirality ('R' or 'S')
            chirality = idx_to_chirality[atom.GetIdx()]
            if chirality == 'S':
                atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            elif chirality == 'R':
                atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    return new_mol



def unassign_chirality_and_delete(rw_mol: Chem.RWMol, atoms_to_delete: list[int]) -> None:
    """
    Remove atoms and ensure they do not define chirality. If they do, drop the
    chirality of the attached chiral center or double bond in a localized way.

    Also handles the special case of deleting substituents from ring heteroatoms
    by allowing implicit hydrogens to be added, preventing sanitization errors.
    """
    bonds_to_neutralize = set()
    centers_to_neutralize = set()
    atoms_to_delete_set = set(atoms_to_delete)
    heteroatoms_to_prep = set()  # Store heteroatoms that need prepping for deletion.
    # Keep track of the original atoms' indices
    for atom in rw_mol.GetAtoms():
        atom.SetIntProp('__ori_atom_index__', atom.GetIdx())
    # --- Stage 1: Discover all stereochemistry and special cases to handle ---
    # Part A: Handle Double Bonds by checking each one's local environment.
    for bond in rw_mol.GetBonds():
        if not (bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetStereo() != Chem.BondStereo.STEREONONE):
            continue
        db_atom1, db_atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
        critical_atoms = {db_atom1.GetIdx(), db_atom2.GetIdx()}
        critical_atoms.update(bond.GetStereoAtoms())
        for atom in [db_atom1, db_atom2]:
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() not in critical_atoms:
                    critical_atoms.add(neighbor.GetIdx())
        if not critical_atoms.isdisjoint(atoms_to_delete_set):
            bonds_to_neutralize.add(bond.GetIdx())
    # Part B: Handle Chiral Centers and the new Heteroatom case "on-the-fly".
    for atom_idx in atoms_to_delete_set:
        atom = rw_mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            # We only care about neighbors that are NOT being deleted themselves.
            if neighbor_idx in atoms_to_delete_set:
                continue
            # Discover surviving aromatic heteroatoms losing a neighbor.
            if neighbor.IsInRing() and neighbor.GetAtomicNum() not in [1, 6]:
                heteroatoms_to_prep.add(neighbor_idx)
            # Discover surviving chiral centers losing a neighbor.
            if neighbor.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
                centers_to_neutralize.add(neighbor_idx)
    # --- Stage 2: Neutralize and Delete (Apply all changes) ---
    # Neutralize affected double bonds and their local single bond directions.
    for bond_idx in bonds_to_neutralize:
        bond_to_neutralize = rw_mol.GetBondWithIdx(bond_idx)
        bond_to_neutralize.SetStereo(Chem.BondStereo.STEREONONE)
        if bond_to_neutralize.HasProp('_CIPCode'):
            bond_to_neutralize.ClearProp('_CIPCode')
        for db_atom in [bond_to_neutralize.GetBeginAtom(), bond_to_neutralize.GetEndAtom()]:
            for attached_bond in db_atom.GetBonds():
                if (attached_bond.GetBondType() == Chem.BondType.SINGLE and
                        attached_bond.GetBondDir() != Chem.BondDir.NONE):
                    is_shared = False
                    other_atom = attached_bond.GetOtherAtom(db_atom)
                    for other_atom_bond in other_atom.GetBonds():
                        if (other_atom_bond.GetStereo() != Chem.BondStereo.STEREONONE and
                                other_atom_bond.GetIdx() != bond_idx):
                            is_shared = True
                            break
                    if not is_shared:
                        attached_bond.SetBondDir(Chem.BondDir.NONE)
    # Neutralize the discovered chiral centers.
    for center_idx in centers_to_neutralize:
        atom = rw_mol.GetAtomWithIdx(center_idx)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        if atom.HasProp('_CIPCode'):
            atom.ClearProp('_CIPCode')
    # Finally, delete the atoms.
    for idx in sorted(list(atoms_to_delete_set), reverse=True):
        rw_mol.RemoveAtom(idx)
    # Fix heteroatoms that have been flagged
    old_to_new_indices_map = {atom.GetIntProp('__ori_atom_index__'): atom.GetIdx()
                              for atom in rw_mol.GetAtoms()}
    for atom_idx in heteroatoms_to_prep:
        atom = rw_mol.GetAtomWithIdx(old_to_new_indices_map[atom_idx])
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(False)
        atom.UpdatePropertyCache(strict=False)
    return


def identify_terminal_atoms(mol: Chem.Mol) -> list[int]:
    """Find all terminal atoms that can be safely deleted.

    :param molecule: molecule to react.
    :return: the list of atom indices that can be safely deleted.
    """
    if not isinstance(mol, Chem.Mol):
        raise ValueError('Molecule is not a valid RDKit Chem.Mol.')
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetIntProp('original_atomid', atom.GetIdx())
    # Precompile the reaction SMARTS for efficiency.
    reactions = []
    for smarts in ['[*:1]-[!#6&D1]>>[*:1]', '[*R:1]-[!#6&D1]>>[*H:1]', '[!#15H:1]=[!#6&D1;!$([*]=[*D4]);!$([*]=[*D3!X4]),!#6&D2H1,2H]>>[*:1]',
                   '[*:1]-[!#6&D1]>>[*:1]', '[*R:1]-[!#6&D1]>>[*H:1]', '[!#15H:1]=[!#6&D1;!$([*]=[*D4]);!$([*]=[*D3!X4]),!#6&D2H1,2H]>>[*:1]']:
    # for smarts in ['[*!R:1][!#6&D1]>>[*:1]', '[*!R:1]=[!#6&D1;!$([*]=[*D4]);!$([*]=[*D3!X4]),!#6&D2H1,2H]>>[*:1]',
    #                '[*!R:1][!#6&D1]>>[*:1]', '[*!R:1]=[!#6&D1;!$([*]=[*D4]);!$([*]=[*D3!X4]),!#6&D2H1,2H]>>[*:1]']:
    # for smarts in ['[#6R:1][!#6&D1,!#6D2H]>>[*:1]', '[*!R:1][!#6D1,!#6D2H]>>[*:1]', '[*R:1][!#6&D1,!#6&D2H]>>[*H:1]', '[*!R:1]=[!#6&D1;!$([*]=[*D4]);!$([*]=[*D3!X4]),!#6&D2H1,2H]>>[*:1]',
    #                '[#6R:1][!#6&D1,!#6D2H]>>[*:1]', '[*!R:1][!#6D1,!#6D2H]>>[*:1]', '[*R:1][!#6&D1,!#6&D2H]>>[*H:1]', '[*!R:1]=[!#6&D1;!$([*]=[*D4]);!$([*]=[*D3!X4]),!#6&D2H1,2H]>>[*:1]']:
        try:
            rxn = AllChem.ReactionFromSmarts(smarts)
            # Ensure reaction is a simple A >> B transformation
            if rxn.GetNumReactantTemplates() == 1 and rxn.GetNumProductTemplates() == 1:
                reactions.append(rxn)
        except Exception as e:
            pass
    if not reactions:
        raise ValueError('No reaction could be parsed from the given list of reaction SMARTS strings.')
    atoms_to_remove = []
    # Work on a copy of the molecule
    current_mol = Chem.Mol(mol)
    # Outer loop: Iterate through each reaction in the master list.
    for rxn_idx, rxn in enumerate(reactions):
        # Middle loop: Apply this single reaction repeatedly up to the limit.
        for i in range(100):
            current_mol_copy = Chem.Mol(current_mol)
            if rxn_idx in [1, 4]:
                current_mol_copy.UpdatePropertyCache()
                Chem.GetSymmSSSR(current_mol_copy)
            outcomes = rxn.RunReactants((current_mol_copy,))
            # If the reaction cannot be applied, the molecule is stable for this reaction.
            if not outcomes:
                break  # Exit the middle 'for' loop.
            product_mol = outcomes[0][0]
            if rxn_idx in [1, 4]:
                product_mol.UpdatePropertyCache()
                Chem.GetSymmSSSR(product_mol)
                product_mol = Chem.RemoveHs(product_mol)
                product_mol = Chem.AddHs(product_mol)
            try:
                with BlockLogs():
                    Chem.SanitizeMol(product_mol)
                if Chem.MolToSmiles(current_mol_copy) != Chem.MolToSmiles(product_mol):
                    current_mol = product_mol
                else:
                    # The product is the same, so we are stable.
                    break  # Exit the middle 'for' loop.
            # Reaction created an invalid molecule; halting for this reaction.
            except Exception as e:
                break  # Exit middle loop on error.
    current_mol.UpdatePropertyCache()
    Chem.GetSymmSSSR(current_mol)
    current_mol = Chem.AddHs(current_mol)
    with BlockLogs():
        Chem.SanitizeMol(current_mol)
    current_mol = Chem.RemoveHs(current_mol)
    # Find atoms of the original molecule that have been deleted
    atom_indices = set(atom.GetIntProp('original_atomid')
                       for atom in Chem.RemoveHs(mol).GetAtoms()).difference(
                                    mol.GetAtomWithIdx(idx).GetIntProp('original_atomid')
                                    for idx in Chem.RemoveHs(mol).GetSubstructMatch(Chem.RemoveHs(current_mol))
                                    )
    return list(atom_indices)


def fix_valence(mol: Chem.Mol) -> Chem.Mol:
    """Ensure the atoms of the molecule have a typical valence and no radical electron
    by adding supplementary hydrogen atoms."""
    rw_mol = Chem.RWMol(mol)
    periodic_table = Chem.GetPeriodicTable()
    for atom in rw_mol.GetAtoms():
        default_valence = periodic_table.GetDefaultValence(atom.GetAtomicNum())
        if atom.GetNumRadicalElectrons() > 0:
            atom.SetNumExplicitHs(atom.GetTotalNumHs() + atom.GetNumRadicalElectrons())
            atom.UpdatePropertyCache()
        if atom.GetTotalValence() < default_valence:
            atom.SetNumExplicitHs(default_valence - atom.GetTotalValence() - atom.GetFormalCharge() + atom.GetNumExplicitHs())
            atom.UpdatePropertyCache()
    mol = rw_mol.GetMol()
    with BlockLogs():
        Chem.SanitizeMol(mol)
    return mol


def prune_hypervalent_atoms(mol: Chem.Mol, pruning: str = 'shortest') -> Chem.Mol:
    """
    Identify hypervalent atoms (valence > 4) and iteratively prune
    substituents with either the lowest CIP priority until all atoms are tetravalent (default) or
    the shortest topological distance.

    :param pruning: one of {'cip', 'shortest'} to prune substituents with lowest CIP priorities or
    with the shortest topological distances of their longest path rooted on the hypervalent atom's
    neighbour and omitting the hypervalent atom.
    """
    if pruning not in ['cip', 'shortest']:
        raise ValueError('pruning must be one of "cip" or "shortest"')
    # Make a hydrogen-deprived copy of the molecule
    current_mol = Chem.Mol(mol)
    while True:
        # Look for a hypervalent atom
        hypervalent_atom_found = None
        for atom in current_mol.GetAtoms():
            if (atom.GetTotalValence() - atom.GetTotalNumHs()) > 4:
                hypervalent_atom_found = atom
                break
        # Exit
        if not hypervalent_atom_found:
            break
        if pruning == 'cip':
            # Assign CIP labels
            center_idx = hypervalent_atom_found.GetIdx()
            neighbors = hypervalent_atom_found.GetNeighbors()
            # # Graceful exit (molecule should have been sanitized)
            # if len(neighbors) <= 4: break
            # Sort neighbors using the custom BFS comparison function.
            custom_key = cmp_to_key(lambda n1, n2: compare_substituents_bfs(current_mol,
                                                                            center_idx,
                                                                            n1.GetIdx(),
                                                                            n2.GetIdx()))
            ranked_neighbors = sorted(neighbors, key=custom_key)
            lowest_priority_neighbor = ranked_neighbors[0]
            # Find the entire fragment attached to the lowest-priority neighbor
            fragment_to_delete = set()
            queue = deque([lowest_priority_neighbor.GetIdx()])
            visited = {center_idx}
            while queue:
                current_idx = queue.popleft()
                if current_idx in visited: continue
                visited.add(current_idx)
                fragment_to_delete.add(current_idx)
                current_atom = current_mol.GetAtomWithIdx(current_idx)
                for neighbor in current_atom.GetNeighbors():
                    queue.append(neighbor.GetIdx())
            # Remove atoms
            rw_mol = Chem.RWMol(current_mol)
            unassign_chirality_and_delete(rw_mol, list(fragment_to_delete))
            current_mol = rw_mol.GetMol()
            with BlockLogs():
                Chem.SanitizeMol(current_mol)
        else:
            center_idx = hypervalent_atom_found.GetIdx()
            neighbors = hypervalent_atom_found.GetNeighbors()
            # Store all longest shortest paths starting from each neighbour
            neighbor_paths = [paths.get_longest_shortest_path_from_atom(current_mol, neighbor.GetIdx(), [center_idx])
                              for neighbor in neighbors]
            # Consider the case in which the path overlaps with the basic scaffold
            bsc_indices = get_basic_scaffold(current_mol, only_atom_indices=True)
            neighbor_paths = [(0 if set(path).isdisjoint(bsc_indices) else 1, path) for path in neighbor_paths]
            # Find the shortest of them
            shortest_path = min(neighbor_paths)
            # Prune either the whole path if it does not overlap the basic scaffold
            # or only the neighbor atom of the center
            current_mol = Chem.RWMol(current_mol)
            if shortest_path[0] == 0:
                unassign_chirality_and_delete(current_mol, shortest_path[1])
            else:
                unassign_chirality_and_delete(current_mol, [shortest_path[1][0]])
            current_mol = current_mol.GetMol()
    return current_mol
