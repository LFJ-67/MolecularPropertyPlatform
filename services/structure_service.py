from io import BytesIO
import base64

from rdkit import Chem
from rdkit.Chem import Draw


from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski


def smiles_to_base64(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    img = Draw.MolToImage(
        mol,
        size=(350, 350)
    )

    buffer = BytesIO()

    img.save(buffer, format="PNG")

    img_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode()

    return img_base64




def molecule_info(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    info = {

        "formula":
            rdMolDescriptors.CalcMolFormula(mol),

        "molwt":
            round(Descriptors.MolWt(mol),2),

        "tpsa":
            round(rdMolDescriptors.CalcTPSA(mol),2),

        "logp":
            round(Crippen.MolLogP(mol),2),

        "hba":
            Lipinski.NumHAcceptors(mol),

        "hbd":
            Lipinski.NumHDonors(mol),

        "rotatable":
            Lipinski.NumRotatableBonds(mol),

        "rings":
            rdMolDescriptors.CalcNumAromaticRings(mol)

    }

    return info

