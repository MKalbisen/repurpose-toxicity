from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from repurpose.data.chemistry import canonicalize


def molecule_svg(smiles):
    canonical = canonicalize(smiles)
    if canonical is None:
        raise ValueError("Geçersiz molekül")
    mol = Chem.MolFromSmiles(canonical)
    drawer = Draw.MolDraw2DSVG(640, 320)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def molecule_3d(smiles, seed=42):
    canonical = canonicalize(smiles)
    if canonical is None:
        return None, "Geçerli SMILES bulunamadı."
    try:
        mol = Chem.AddHs(Chem.MolFromSmiles(canonical))
        if mol.GetNumAtoms() > 250:
            return None, "Bu molekül etkileşimli 3D boyut sınırını aşıyor."
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None, "3D konformasyon üretilemedi; 2D yapı kullanılabilir."
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=100)
        return Chem.MolToMolBlock(mol), None
    except (ValueError, RuntimeError) as e:
        return None, f"3D yapı üretilemedi: {e}"
