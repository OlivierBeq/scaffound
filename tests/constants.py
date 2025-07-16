# -*- coding: utf-8 -*-


"""Constants for scaffound unittesting."""


import os

import pandas as pd


SAMPLES_FOLDER = os.path.realpath(os.path.dirname(__file__))

# Table S2 from the original article, adapted to RDKit (see MODIFICATIONS.txt)
COX2_SCAFFOLDS = pd.read_csv(os.path.join(SAMPLES_FOLDER, 'cox2_816_inhibitors.txt'), sep='\t')
COX2_SCAFFOLDS = COX2_SCAFFOLDS.drop(columns=[col for col in COX2_SCAFFOLDS.columns if col.endswith('inchikey')])
COX2_SCAFFOLDS = COX2_SCAFFOLDS.set_axis(map(str.lower, COX2_SCAFFOLDS), axis='columns')

# Table S2 from the original article, adapted to the improved longest path algorithm (see MODIFICATIONS.txt)
COX2_SCAFFOLDS_EXT = pd.read_csv(os.path.join(SAMPLES_FOLDER, 'cox2_816_inhibitors_adapted_lsp.txt'), sep='\t')
COX2_SCAFFOLDS_EXT = COX2_SCAFFOLDS_EXT.drop(columns=[col for col in COX2_SCAFFOLDS_EXT.columns if col.endswith('inchikey')])
COX2_SCAFFOLDS_EXT = COX2_SCAFFOLDS_EXT.set_axis(map(str.lower, COX2_SCAFFOLDS_EXT), axis='columns')

# Longest paths with identical lengths
# COX2_SCAFFOLDS = COX2_SCAFFOLDS.drop(index=[554])

NEW_SCAFFFOLDS = pd.read_csv(os.path.join(SAMPLES_FOLDER, 'additional_scaffolds.txt'), sep='\t')
NEW_SCAFFFOLDS = NEW_SCAFFFOLDS.drop(columns=[col for col in NEW_SCAFFFOLDS.columns if col.endswith('inchikey')])
NEW_SCAFFFOLDS = NEW_SCAFFFOLDS.set_axis(map(str.lower, NEW_SCAFFFOLDS), axis='columns')
