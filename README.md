
<p align="center">
  <img src="https://github.com/OlivierBeq/scaffound/raw/master/graphics/logo/scaffound-logo-dark-50percent.png" alt="scaffound logo" width="300px"/>
</p>

--------------------------------------------------------------------------------

A Python library for extracting multiple types of molecular scaffolds, frameworks, and wireframes.


`scaffound` provides a hierarchical approach to molecular decomposition derived from **[[1]](https://doi.org/10.1186/s13321-021-00526-y)**, allowing for a detailed analysis of chemical structures beyond the traditional Bemis-Murcko scaffold.

`scaffound` is an extended implementation of Dompé's *Molecular Anatomy*  to identify different types of molecular scaffolds, frameworks and wireframes.

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/>
  </a>
  <a href="https://doi.org/10.5281/zenodo.16743086">
    <img src="https://zenodo.org/badge/1021088731.svg" alt="DOI"/>
  </a>
  <img src="https://badge.fury.io/py/scaffound.svg" alt="PyPI version"/>
  <img src="https://img.shields.io/pypi/pyversions/scaffound" alt="PyPI - Python Version"/>
</p>

# Core concepts ⚛️

The library extracts three main types of scaffolds:

- **Basic Scaffold**: The core ring systems and their linkers.
- **Decorated Scaffold**: The basic scaffold plus all heteroatoms directly attached to it by unsaturated bonds.
- **Augmented Scaffold**: The decorated scaffold plus all atoms belonging to the longest carbon chain (including substituents and side chains).

These scaffolds can be further abstracted into:
- **Frameworks**: Scaffolds made generic by replacing all heteroatoms with carbons.
- **Wireframes**: Scaffolds made both saturated (all bonds replaced with single bonds) and generic.


<p align="center">
  <img src="https://github.com/OlivierBeq/scaffound/raw/master/graphics/hierarchy/scaffolds.svg" alt="scaffound hierarchy" width="800px"/>
</p>

## ⚙️ A Note on Augmented Scaffolds
The seminal algorithm for determining the augmented scaffold relies on identifying the longest path within the molecular graph.
<br/>However, the original description's method does not determine a unique path when multiple paths of the same maximum length exist and chooses one among all solutions.

> "[...] the longest atom chain, considering also substitutions, is retained but all
> terminal non-carbon atoms, belonging to side chains, are iteratively pruned (Augmented Scaffold)."
>
> "[...] three paths can be identified two of them are the longest with the same length and the first identified is retained"

This means that multiple valid paths could be chosen for the same molecule, each resulting in a different augmented scaffold.


Consequently, while `scaffound` strictly adheres to the published logic, its implementation may identify different (yet equally valid) longest paths than those in the original work.
<br/>This can lead to variations in the resulting augmented scaffolds compared to the examples in the source publication.

To address this ambiguity, `scaffound` also **implements its own deterministic canonical longest path algorithm**.
This ensures a single deterministic outcome.

# Installation 🪄

```bash
pip install scaffound
```

# Geting started 🚀

```python
# A simple usage example
from rdkit import Chem
from scaffound import MolecularAnatomy

# Create an RDKit molecule object
mol = Chem.MolFromSmiles('O=C(c1ccccc1)N1CCN(c2ccccc2)CC1')


# Analyze the molecule
anatomy = MolecularAnatomy(mol)

# Access different scaffold/framework/wireframe types
basic_scaffold = anatomy.basic_scaffold
decorated_framework = anatomy.decorated_framework
augmented_wireframe = anatomy.augmented_wireframe

# You can now work with these new molecule objects
print(Chem.MolToSmiles(basic_scaffold))
# Output: c1ccc(C2CCN(c3ccccc3)CC2)cc1
```

The `MolecularAnatomy` object also decomposes the molecule's generic and saturated graphs into all the scaffolds, frameworks and wireframes mentioned above.<br/>

<p align="center">
  <img src="https://github.com/OlivierBeq/scaffound/raw/master/graphics/hierarchy/other_graphs.svg" alt="scaffound hierarchy" width="900px"/>
</p>

An entire decomposition can be accessed using its `to_dict()` or `to_pandas()` methods.<br/>

Mind you that some decompositions of the original, generic, and saturated graphs are identical (see [decomposition_equivalence.ipynb](https://github.com/OlivierBeq/Scaffound/blob/master/docs/decomposition_equivalence.ipynb)).<br/>
For instance:
- the basic framework of the original graph is the same as the basic scaffold of the generic graph,
- the basic wireframe of the original graph is the same as the basic wireframe of the generic graph,
- the basic framework of the saturated graph is the same as the decorated_framework of the saturated graph.


# Advanced usage 💪

If performance is needed, one can use functions to access only the type of scaffold/framework/wireframe needed (since the `MolecularAnatomy` decomposes a molecule ahead of time into all the possible scaffolds). 


```python
from scaffound import (get_generic_graph, # All heteroatoms replaced by carbons
                       get_saturated_graph, # All bonds replaced by single bonds
                       # Scaffold types
                       get_basic_scaffold, get_decorated_scaffold, get_augmented_scaffold,
                       # Framework types
                       get_basic_framework, get_decorated_framework, get_augmented_framework,
                       # Wireframe types
                       get_basic_wireframe, get_decorated_wireframe, get_augmented_wireframe)
```

Furthermore, one can deactivate `scaffound`'s deterministic longest path algorithm or fine-tune its behavior.
The `MinMaxShortestPathOptions` class gives you full control over the tie-breaking cascade used to select the canonical path.
For example, you can prioritize paths with fewer rings, larger atomic numbers, or different topologies.

To revert to the original, non-deterministic algorithm from the paper:

```python
from scaffound import MinMaxShortestPathOptions

opts = MinMaxShortestPathOptions(original_algorithm=True)

# These functions will now use the original algorithm
MolecularAnatomy(mol, opts=pts)
get_augmented_scaffold(mol, opts=opts)
get_augmented_framework(mol, opts=opts)
get_augmented_wireframe(mol, opts=opts)
```


# Validation ✅

This library has been rigorously tested against the exemplary file from the seminal scientific article that introduced these concepts.
<br/>The reference data has been corrected within this repository to ensure it aligns 100% with the paper's detailed algorithm and figures, providing a reliable and verified tool (see [tests/MODIFICATIONS.txt](https://github.com/OlivierBeq/Scaffound/blob/master/tests/MODIFICATIONS.txt)).
<br/>An adapted version of this reference data is also provided ([tests/cox2_816_inhibitors_adapted_lsp.txt](https://github.com/OlivierBeq/Scaffound/blob/master/tests/cox2_816_inhibitors_adapted_lsp.txt)) to reflect the results of `scaffound`'s deterministic longest path algorithm, which is also described in [tests/MODIFICATIONS.txt](https://github.com/OlivierBeq/Scaffound/blob/master/tests/MODIFICATIONS.txt).

# References 📜

- [1] Manelfi, C., Gemei, M., Talarico, C. et al.<br/>
      “Molecular Anatomy”: a new multi-dimensional hierarchical scaffold analysis tool.<br/>
      J Cheminform 13, 54 (2021).
      https://doi.org/10.1186/s13321-021-00526-y


- [2] Rosario Beccari, A. (2021)
      Molecular_Anatomy [Computer software].</br>
      https://github.com/andreabeccari/Molecular_Anatomy
