import torch
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdmolops
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem.GraphDescriptors import BalabanJ, BertzCT, Chi0n, Chi1n, Kappa1, Kappa2
from rdkit.Chem.rdchem import HybridizationType
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------- 查询表 -----------
ELECTRONEGATIVITY = {1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66, 44: 2.20, 77: 2.20}
POLARIZABILITY = {1: 0.67, 5: 0.80, 6: 1.76, 7: 1.10, 8: 0.80, 9: 0.56, 15: 3.63, 16: 2.90, 17: 2.18, 35: 3.05, 53: 4.70, 44: 6.10, 77: 7.00}
VALENCE_ELECTRONS = {1: 1, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 15: 5, 16: 6, 17: 7, 35: 7, 53: 7, 44: 8, 77: 9}

# ----------- 辅助函数 -----------
def safe_scalar(val):
    if val is None:
        return 0.0
    try:
        return float(val)
    except Exception:
        pass
    try:
        arr = list(val)
        flat = []
        for v in arr:
            try:
                flat.append(float(v))
            except Exception:
                continue
        if len(flat) > 0:
            return float(np.mean(flat))
        else:
            return 0.0
    except Exception:
        return 0.0

def CalcNumAromaticAtoms(mol):
    return sum(atom.GetIsAromatic() for atom in mol.GetAtoms())

def calc_num_conjugated_bonds(mol):
    return sum(1 for bond in mol.GetBonds() if bond.GetIsConjugated())

def calculate_aromatic_distances(mol):
    num_atoms = mol.GetNumAtoms()
    aromatic_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
    if not aromatic_atoms:
        return [10.0] * num_atoms
    try:
        dist_mat = Chem.rdmolops.GetDistanceMatrix(mol)
    except:
        return [10.0] * num_atoms
    return [float(min(dist_mat[i, j] for j in aromatic_atoms)) for i in range(num_atoms)]

# ----------- 特征提取函数 -----------
def compute_mol_global_features(mol):
    features = [
        safe_scalar(rdMolDescriptors.CalcNumAromaticRings(mol)),
        safe_scalar(CalcNumAromaticAtoms(mol)),
        safe_scalar(calc_num_conjugated_bonds(mol)),
        safe_scalar(Chem.GetSSSR(mol)),
        safe_scalar(rdMolDescriptors.CalcNumHBD(mol)),
        safe_scalar(rdMolDescriptors.CalcNumHBA(mol)),
        safe_scalar(rdMolDescriptors.CalcTPSA(mol)),
        safe_scalar(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        safe_scalar(BalabanJ(mol)),
        safe_scalar(BertzCT(mol)),
        safe_scalar(Chi0n(mol)),
        safe_scalar(Chi1n(mol)),
        safe_scalar(Kappa1(mol)),
        safe_scalar(Kappa2(mol))
    ]

    try:
        ComputeGasteigerCharges(mol)
        charges = [float(atom.GetProp('_GasteigerCharge'))
                   for atom in mol.GetAtoms()
                   if atom.HasProp('_GasteigerCharge')]
        if charges:
            features.extend([
                float(np.mean(charges)),
                float(np.std(charges)),
                float(np.min(charges)),
                float(np.max(charges))
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
    except Exception:
        features.extend([0.0, 0.0, 0.0, 0.0])

    return torch.tensor(features, dtype=torch.float)

def get_morgan_fingerprint(mol, radius=2, size=2048):
    fp = AllChem.GetMorganGenerator(radius=radius, fpSize=size).GetFingerprint(mol)
    arr = np.zeros((size,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return torch.tensor(arr, dtype=torch.float).unsqueeze(0)

def atom_features(atom, mol):
    an = atom.GetAtomicNum()
    features = [
        an,
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        atom.GetImplicitValence(),
        1.0 if atom.GetIsAromatic() else 0.0,
        1.0 if atom.IsInRing() else 0.0,
        atom.GetMass()
    ]

    hyb = atom.GetHybridization()
    hyb_map = {HybridizationType.SP: [1,0,0], HybridizationType.SP2: [0,1,0], HybridizationType.SP3: [0,0,1]}
    features.extend(hyb_map.get(hyb, [0,0,0]))

    try:
        q = float(atom.GetProp("_GasteigerCharge"))
        q = 0.0 if np.isnan(q) else q
    except:
        q = 0.0
    features.append(q)

    features.append(ELECTRONEGATIVITY.get(an, 2.0))
    features.append(POLARIZABILITY.get(an, 1.5))
    features.append(VALENCE_ELECTRONS.get(an, 4))

    neighbors = atom.GetNeighbors()
    neighbor_elec = [ELECTRONEGATIVITY.get(n.GetAtomicNum(), 2.0) for n in neighbors]
    avg_elec = np.mean(neighbor_elec) if neighbor_elec else ELECTRONEGATIVITY.get(an, 2.0)
    features.append(avg_elec)

    dists = calculate_aromatic_distances(mol)
    features.append(dists[atom.GetIdx()])

    features.append(1.0 if an > 10 else 0.0)
    is_pi = 1.0 if (atom.GetIsAromatic() or hyb == HybridizationType.SP2) else 0.0
    features.append(is_pi)

    return torch.tensor(features, dtype=torch.float)

def bond_features(bond):
    bt = bond.GetBondType()
    return torch.tensor([
        1.0 if bt == Chem.rdchem.BondType.SINGLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.DOUBLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.TRIPLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.AROMATIC else 0.0,
        1.0 if bond.IsInRing() else 0.0
    ], dtype=torch.float)

# ----------- 构建图数据函数 -----------
def smiles_to_graph_data(smiles, label=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = Chem.AddHs(mol)
        Chem.SanitizeMol(mol)
        ComputeGasteigerCharges(mol)
        mol = Chem.RemoveHs(mol)
    except:
        pass

    if mol.GetNumAtoms() == 0:
        return None

    x = torch.stack([atom_features(atom, mol) for atom in mol.GetAtoms()])

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
        efeat = bond_features(bond)
        edge_attr.append(efeat)
        edge_attr.append(efeat)
    edge_index = torch.tensor(edge_index).t().contiguous() if edge_index else torch.zeros((2,0), dtype=torch.long)
    edge_attr = torch.stack(edge_attr) if edge_attr else torch.zeros((0,5), dtype=torch.float)

    mol_feats = compute_mol_global_features(mol)
    fp = get_morgan_fingerprint(mol, size=2048)

    y = torch.tensor([float(label)], dtype=torch.float) if label is not None else None

    from torch_geometric.data import Data
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.mol_feats = mol_feats.unsqueeze(0)
    data.fp = fp
    return data

# ----------- 模型定义 -----------
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool


class StableResidualGAT(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, num_layers=4, heads=4, dropout_p=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.res_projs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_p)
        self.act = nn.ELU()

        # 第一层
        self.convs.append(GATConv(in_channels, hidden_dim // heads, heads=heads, concat=True, dropout=dropout_p))
        self.res_projs.append(nn.Linear(in_channels, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))

        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True, dropout=dropout_p))
            self.res_projs.append(nn.Identity())
            self.norms.append(nn.LayerNorm(hidden_dim))

        # 最后一层
        self.convs.append(GATConv(hidden_dim, out_channels, heads=1, concat=False, dropout=dropout_p))
        self.res_projs.append(nn.Linear(hidden_dim, out_channels))
        self.norms.append(nn.LayerNorm(out_channels))

    def forward(self, x, edge_index, batch):
        for i, (conv, res_proj, norm) in enumerate(zip(self.convs, self.res_projs, self.norms)):
            res = x
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

            res_proj_out = res_proj(res)
            res_proj_out = torch.nan_to_num(res_proj_out, nan=0.0, posinf=1e6, neginf=-1e6)

            if res_proj_out.shape != x.shape:
                if res_proj_out.shape[1] < x.shape[1]:
                    pad = torch.zeros((res_proj_out.shape[0], x.shape[1] - res_proj_out.shape[1]),
                                      device=res_proj_out.device)
                    res_proj_out = torch.cat([res_proj_out, pad], dim=1)
                elif res_proj_out.shape[1] > x.shape[1]:
                    res_proj_out = res_proj_out[:, :x.shape[1]]

            x = x + res_proj_out
            x = norm(x)
            x = torch.clamp(x, -1e3, 1e3)
            x = self.act(x) if i != len(self.convs) - 1 else x

            if torch.isnan(x).any() or torch.isinf(x).any():
                x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        return global_mean_pool(x, batch)


class StableGNN(nn.Module):
    def __init__(self, node_feat_dim, fp_size=2048, mol_feat_dim=18, hidden_dim=64, out_dim=1):
        super().__init__()
        self.gat = StableResidualGAT(node_feat_dim, hidden_dim, hidden_dim, num_layers=4, heads=4, dropout_p=0.3)

        self.fp_proj = nn.Sequential(
            nn.Linear(fp_size, 256),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64)
        )
        self.fp_norm = nn.LayerNorm(64)

        self.mol_proj = nn.Sequential(
            nn.Linear(mol_feat_dim, 64),
            nn.ELU(),
            nn.Dropout(0.3)
        )
        self.mol_norm = nn.LayerNorm(64)

        self.combined_norm = nn.LayerNorm(hidden_dim + 64 + 64)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 64 + 64, 128),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x, edge_index, batch, fp, mol_feats):
        graph_emb = self.gat(x, edge_index, batch)
        fp_emb = self.fp_proj(fp.squeeze(1))
        fp_emb = self.fp_norm(fp_emb)
        mol_emb = self.mol_proj(mol_feats.squeeze(1))
        mol_emb = self.mol_norm(mol_emb)

        combined = torch.cat([graph_emb, fp_emb, mol_emb], dim=1)
        combined = self.combined_norm(combined)
        combined = torch.clamp(combined, -1e3, 1e3)

        output = self.classifier(combined).squeeze(-1)
        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)

        return output


# ----------- 预测函数 -----------
@torch.no_grad()
def predict_new_smiles(
        smiles: str,
        model,
        scaler
):

    data = smiles_to_graph_data(
        smiles,
        label=0.0
    )

    if data is None:
        return None

    loader = DataLoader(
        [data],
        batch_size=1,
        shuffle=False
    )

    for batch in loader:

        batch = batch.to(DEVICE)

        out = model(

            batch.x,

            batch.edge_index,

            batch.batch,

            batch.fp,

            batch.mol_feats

        )

        out = out.view(-1).cpu().numpy()

        pred = scaler.inverse_transform(
            out.reshape(-1,1)
        )

        return float(pred.ravel()[0])

    return None

# ----------- 示例预测 -----------
if __name__ == "__main__":
    test_smiles = "C1(N=CN2)=C2C=CC=C1"
    pred_value = predict_new_smiles(test_smiles)
    if pred_value is not None:
        print(f"Predicted E_T1 (kcal/mol): {pred_value:.4f}")
    else:
        print("SMILES 解析失败。")