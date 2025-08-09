# -*- coding: utf-8 -*-


"""Helper functions to obtain different flavours of scaffolds."""

from collections import deque
from functools import cmp_to_key

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.rdBase import BlockLogs

from .cip import compare_substituents_bfs
from . import scaffolds
from . import paths



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
            bsc_indices = scaffolds.get_basic_scaffold(current_mol, only_atom_indices=True)
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
