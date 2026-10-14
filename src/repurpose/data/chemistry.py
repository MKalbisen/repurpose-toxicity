"""Shared standardization for training, mapping, inference and rendering."""

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
TAUTOMERS = rdMolStandardize.TautomerEnumerator()


def canonicalize(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or not mol.GetNumAtoms():
            return None
        # Preserve stereochemistry and charge. Salt fragments are retained, logged and
        # never silently stripped. Tautomer canonicalization is a split identity only.
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except (ValueError, RuntimeError):
        return None


def fingerprint(smiles):
    canonical = canonicalize(smiles)
    if canonical is None:
        raise ValueError(f"Geçersiz SMILES: {smiles!r}")
    return GENERATOR.GetFingerprintAsNumPy(Chem.MolFromSmiles(canonical)).astype(np.uint8)


def scaffold(smiles):
    mol = Chem.MolFromSmiles(canonicalize(smiles))
    # Strip salts only for group identity and ignore stereo/tautomers for leakage audit.
    parent = rdMolStandardize.FragmentParent(mol)
    parent = TAUTOMERS.Canonicalize(parent)
    key = MurckoScaffold.MurckoScaffoldSmiles(mol=parent, includeChirality=False)
    # All acyclic compounds share the empty scaffold; keep them in one split.
    return key or "__ACYCLIC__"


def metadata():
    return {
        "rdkit_version": rdBase.rdkitVersion,
        "radius": 2,
        "n_bits": 1024,
        "binary": True,
        "include_chirality": False,
        "canonical_smiles": "isomeric, charge/stereo/salt retained",
        "split_identity": "fragment parent + canonical tautomer + achiral Murcko; acyclic together",
    }
