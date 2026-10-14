import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from xgboost import DMatrix

from repurpose.data.chemistry import canonicalize, fingerprint


def bit_contributions(classifier, smiles, top=12):
    x = fingerprint(smiles).reshape(1, -1).astype(np.float32)
    # Native TreeSHAP contributions, in raw margin/log-odds units, including bias.
    contributions = classifier.get_booster().predict(DMatrix(x), pred_contribs=True)[0]
    mol = Chem.MolFromSmiles(canonicalize(smiles))
    info = rdFingerprintGenerator.AdditionalOutput()
    info.AllocateBitInfoMap()
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    generator.GetFingerprint(mol, additionalOutput=info)
    bit_info = info.GetBitInfoMap()
    order = np.argsort(-np.abs(contributions[:-1]))[:top]
    rows = [
        {
            "bit": int(i),
            "present": bool(x[0, i]),
            "log_odds_contribution": float(contributions[i]),
            "atom_radius_environments": str(bit_info.get(int(i), ())),
            "collision_note": "1024-bit hash may represent multiple environments",
        }
        for i in order
    ]
    return pd.DataFrame(rows), {
        "bias": float(contributions[-1]),
        "sum_log_odds": float(contributions.sum()),
        "raw_margin": float(classifier.get_booster().predict(DMatrix(x), output_margin=True)[0]),
    }
