# -*- coding: utf-8 -*-


"""Custom implementation of the Canh-Ingold-Prelog priority rules."""


from collections import deque

from rdkit import Chem


def get_atom_descriptor(atom: Chem.Atom,
                        is_ghost: bool,
                        previous_atom: Chem.Atom | None = None) -> tuple:
    """
    Generates a sortable descriptor tuple for an atom based on fundamental CIP rules,
    reading raw stereo flags instead of relying on pre-computed properties.
    The tuple is structured to be compared in stages: (non-stereo_parts, stereo_parts).

    A numerically SMALLER tuple represents a HIGHER priority.
    """
    # Rule 1a & 1b: Atomic number (desc) then mass (desc).
    # We negate them so higher values result in a lower sort value (higher priority).
    atomic_num = -atom.GetAtomicNum()
    mass = -atom.GetMass()
    # Rule 2: Stereochemistry. R/Z have priority over S/E.
    # The order of these checks in the tuple establishes their priority.
    # Check for R/S stereochemistry at the atom center.
    chiral_type = 2  # Default: no specified chirality
    chiral_tag = atom.GetChiralTag()
    if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:  # R
        chiral_type = 0
    elif chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:  # S
        chiral_type = 1
    # Check for E/Z stereochemistry if the atom is part of a double bond.
    bond_stereo_type = 2  # Default: not part of a stereo double bond
    if previous_atom:
        bond: Chem.Bond = atom.GetOwningMol().GetBondBetweenAtoms(atom.GetIdx(), previous_atom.GetIdx())
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            stereo_flag = bond.GetStereo()
            if stereo_flag == Chem.BondStereo.STEREOZ or stereo_flag == Chem.BondStereo.STEREOCIS:
                bond_stereo_type = 0
            elif stereo_flag == Chem.BondStereo.STEREOE or stereo_flag == Chem.BondStereo.STEREOTRANS:
                bond_stereo_type = 1
    # Rule 3: Real atoms (0) have priority over ghost atoms (1).
    ghost_flag = 1 if is_ghost else 0
    # Separate non-stereo part from the stereo part
    return ((ghost_flag, atomic_num, mass), (chiral_type, bond_stereo_type))


def compare_substituents_bfs(mol: Chem.Mol, center_idx: int, neighbor1_idx: int, neighbor2_idx: int) -> int:
    """
    Compares two substituents of a central atom using a Breadth-First Search
    that implements the primary CIP rules by reading raw atom/bond properties.
    """
    # Initialize the stacks with (neighbor1_idx, previous_atom1, is_ghost) and
    # (neighbor2_idx, previous_atom1, is_ghost) for each substituent, with previous
    # atoms set to None for the first iteration
    q1, q2 = deque([(neighbor1_idx, None, False)]), deque([(neighbor2_idx, None, False)])
    # Keep track of the visited atoms to prevent cycles and backtracking
    visited1, visited2 = {center_idx, neighbor1_idx}, {center_idx, neighbor2_idx}
    # BFS algorithm
    while q1 and q2:
        sphere_size1, sphere_size2 = len(q1), len(q2)
        sphere_non_stereo1, sphere_non_stereo2 = [], []
        sphere_stereo1, sphere_stereo2 = [], []
        # Process the full "sphere" of atoms for the first substituent
        for _ in range(sphere_size1):
            curr_idx, prev_idx, is_ghost = q1.popleft()
            atom = mol.GetAtomWithIdx(curr_idx)
            prev_atom = mol.GetAtomWithIdx(prev_idx) if prev_idx else None
            # Add priority descriptors
            non_stereo, stereo = get_atom_descriptor(atom, is_ghost, prev_atom)
            sphere_non_stereo1.append(non_stereo)
            sphere_stereo1.append(stereo)
            if is_ghost: continue
            # Expand the sphere
            for bond in atom.GetBonds():
                other_atom = bond.GetOtherAtom(atom)
                other_idx = other_atom.GetIdx()
                if other_idx not in visited1:
                    visited1.add(other_idx)
                    # Add new atom to stack
                    q1.append((other_idx, curr_idx, False))
                    # Ensure bond multiplicity creates ghost atoms
                    bond_type = bond.GetBondTypeAsDouble()
                    if bond_type >= 2: q1.append((other_idx, curr_idx, True))
                    if bond_type >= 3: q1.append((other_idx, curr_idx, True))
        # Now for the second substituent
        for _ in range(sphere_size2):
            curr_idx, prev_idx, is_ghost = q2.popleft()
            atom = mol.GetAtomWithIdx(curr_idx)
            prev_atom = mol.GetAtomWithIdx(prev_idx) if prev_idx else None
            non_stereo, stereo = get_atom_descriptor(atom, is_ghost, prev_atom)
            sphere_non_stereo2.append(non_stereo)
            sphere_stereo2.append(stereo)
            if is_ghost: continue
            for bond in atom.GetBonds():
                other_atom = bond.GetOtherAtom(atom)
                other_idx = other_atom.GetIdx()
                if other_idx not in visited2:
                    visited2.add(other_idx)
                    q2.append((other_idx, curr_idx, False))
                    bond_type = bond.GetBondTypeAsDouble()
                    if bond_type >= 2: q2.append((other_idx, curr_idx, True))
                    if bond_type >= 3: q2.append((other_idx, curr_idx, True))
        # Sort the non-stereochemical properties
        sphere_non_stereo1.sort()
        sphere_non_stereo2.sort()
        # Compare them
        if sphere_non_stereo1 < sphere_non_stereo2: return 1  # Sub2 is lower priority
        if sphere_non_stereo1 > sphere_non_stereo2: return -1  # Sub1 is lower priority
        # Tie-breaker: Use Rule 4b for stereochemistry (like elements have precedence over unlike elements)
        # Filter out non-stereo descriptors (where type is 2)
        tags1 = sorted([tag for tag_pair in sphere_stereo1 for tag in tag_pair if tag != 2])
        tags2 = sorted([tag for tag_pair in sphere_stereo2 for tag in tag_pair if tag != 2])
        # Check for "likeness" (all elements in the list are the same)
        is_like1 = len(set(tags1)) <= 1
        is_like2 = len(set(tags2)) <= 1
        # like > unlike
        if is_like1 and not is_like2: return 1  # Sub2 is lower priority
        if not is_like1 and is_like2: return -1  # Sub1 is lower priority
        # If both are 'like' (e.g., RR vs SS) or both are 'unlike' (e.g., RS vs SR),
        # compare them lexicographically. Since R=0 and S=1, this automatically
        # prioritizes RR > SS and RS > SR, which is the correct final tie-breaker.
        if tags1 < tags2: return 1
        if tags1 > tags2: return -1
        # If spheres are completely identical, continue to the next sphere.
    if not q1 and q2: return -1  # Path 1 is shorter -> lower priority
    if not q2 and q1: return 1  # Path 2 is shorter -> lower priority
    return 0  # They are identical
