import os
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem


# MAX_ATOMNUM = 128
# # NOTE: Padding -1 did not work well.
# ATOM_PAD_VAL= 0


def get_atoms_mask(mols, MAX_ATOMNUM):
    """
    Get the mask of atoms in the molecule.
    """
    n_mols = len(mols)
    mask = np.ones((n_mols, MAX_ATOMNUM), dtype=int)
    for i, mol in enumerate(tqdm(mols)):
        if mol is None:
            num_now = 0
        else:
            num_now = mol.GetNumAtoms()
        num_now = num_now if num_now <= MAX_ATOMNUM else MAX_ATOMNUM
        if num_now > 0:
            mask[i, :num_now] = 0
    return mask




def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set{allowable_set}:")
    return list(map(lambda s: int(x == s), allowable_set))




def one_of_k_encoding_unk(x, allowable_set):
    # NOTE: Allow unknown type
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: int(x == s), allowable_set))




def calculate_atoms_features(mols, MAX_ATOMNUM, ATOM_PAD_VAL):
    """""
    Calculate 30 types atom features.
    """""
    atoms_features = []

    ATTRIBUTE_LEN = 30
    n_mols = len(mols)
    features = np.full((n_mols, MAX_ATOMNUM, ATTRIBUTE_LEN), ATOM_PAD_VAL, dtype=float)
    for i, mol in enumerate(tqdm(mols)):
        atoms = [] if mol is None else list(mol.GetAtoms())
        for j, atom in enumerate(atoms[:MAX_ATOMNUM]):
            attributes = []
            # Symbol: ['C', 'N', 'O', 'F', 'Br', 'S', 'Cl']
            # 8 dims
            attributes += one_of_k_encoding_unk(
                atom.GetSymbol(),
                ['C', 'N', 'O', 'F', 'Br', 'S', 'Cl', 'Other']
            )
            # CHANGED len(GetNeighbors) -> GetDegree
            # Degree: [1, 2, 3, 4]
            # 5 dims
            attributes += one_of_k_encoding_unk(
                atom.GetDegree,
                [1, 2, 3, 4, 'Other']
            )
            # TotalNumHs: [0, 1, 2, 3]
            # 5 dims
            attributes += one_of_k_encoding_unk(
                atom.GetTotalNumHs(),
                [0, 1, 2, 3, 'Other']
            )
            # FormalCharge: [0, 1, -1]
            # 4 dims
            attributes += one_of_k_encoding_unk(
                atom.GetFormalCharge(),
                [-1, 0, 1, 'Other']
            )
            # Hybridization: [rdkit.Chem.rdchem.HybridizationType.SP2, rdkit.Chem.rdchem.HybridizationType.SP3]
            # 3 dims
            attributes += one_of_k_encoding_unk(
                atom.GetHybridization(),
                [Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, 'Other']
            )
            # ChiralTag: [rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED, rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW]
            # 3 dims
            attributes += one_of_k_encoding_unk(
                atom.GetChiralTag(),
                [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.rdchem.ChiralType.CHI_UNSPECIFIED]
            )
            # 2 dims
            attributes.append(int(atom.IsInRing()))
            attributes.append(int(atom.GetIsAromatic()))

            # write attributes into the preallocated array
            try:
                features[i, j, :] = np.array(attributes, dtype=float)
            except Exception:
                # fallback: if shapes mismatch, pad/truncate attributes
                arr = np.array(attributes, dtype=float)
                if arr.size >= ATTRIBUTE_LEN:
                    features[i, j, :] = arr[:ATTRIBUTE_LEN]
                else:
                    tmp = np.full((ATTRIBUTE_LEN,), ATOM_PAD_VAL, dtype=float)
                    tmp[:arr.size] = arr
                    features[i, j, :] = tmp
    return features




def calculate_graph_distance_matrix(mols, MAX_ATOMNUM, ATOM_PAD_VAL):
    """
    Calculate atoms pairwise graph distance matrix of molecule.
    """
    n_mols = len(mols)
    distance_graph = np.full((n_mols, MAX_ATOMNUM, MAX_ATOMNUM), ATOM_PAD_VAL, dtype=float)
    for i, mol in enumerate(tqdm(mols)):
        if mol is None:
            continue
        num_now = mol.GetNumAtoms()
        full_mat = Chem.rdmolops.GetDistanceMatrix(mol)
        import numpy as _np
        mat = _np.array(full_mat)
        if mat.size == 0:
            continue
        if num_now >= MAX_ATOMNUM:
            distance_graph[i] = mat[:MAX_ATOMNUM, :MAX_ATOMNUM]
        else:
            distance_graph[i, :num_now, :num_now] = mat
    return distance_graph




def calculate_conf_distance_matrix(mols, MAX_ATOMNUM, ATOM_PAD_VAL):
    """
    Calculate atoms pairwise 3D distance matrix of molecule.
    """
    n_mols = len(mols)
    distance_conf = np.full((n_mols, MAX_ATOMNUM, MAX_ATOMNUM), ATOM_PAD_VAL, dtype=float)
    import numpy as _np
    for i, mol in enumerate(tqdm(mols)):
        if mol is None:
            continue
        num_now = mol.GetNumAtoms()
        try:
            full_mat = AllChem.Get3DDistanceMatrix(mol)
            mat = _np.array(full_mat)
            if mat.size == 0:
                # try compute from conformer if available
                if mol.GetNumConformers() > 0:
                    conf = mol.GetConformer(0)
                    coords = _np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
                    d = _np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
                    mat = d
                else:
                    continue
        except Exception:
            if mol.GetNumConformers() > 0:
                conf = mol.GetConformer(0)
                coords = _np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
                mat = _np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
            else:
                continue

        if num_now >= MAX_ATOMNUM:
            distance_conf[i] = mat[:MAX_ATOMNUM, :MAX_ATOMNUM]
        else:
            distance_conf[i, :num_now, :num_now] = mat

    # NOTE: Round to 4 decimal places
    return np.round(distance_conf, 4)




def calculate_bond_type_matrix(mols, MAX_ATOMNUM):
    """
    Calculate atoms pairwise bond type matrix of molecule.
    """
    n_mols = len(mols)
    bond_types = np.zeros((n_mols, MAX_ATOMNUM, MAX_ATOMNUM), dtype=float)
    weight = [1, 2, 3, 1.5]
    for idx, mol in enumerate(tqdm(mols)):
        if mol is None:
            continue
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            if a1 >= MAX_ATOMNUM or a2 >= MAX_ATOMNUM:
                continue
            bt = bond.GetBondType()
            bond_feats = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC]
            for i, m in enumerate(bond_feats):
                if m == True and i != 0:
                    b = weight[i]
                elif m == True and i == 0:
                    if bond.GetIsConjugated() == True:
                        b = 1.4
                    else:
                        b = 1
                else:
                    continue
            bond_types[idx, a1, a2] = b
            bond_types[idx, a2, a1] = b

    return bond_types




def generate_atom_input(config, df, df_enu, mols, folder_path, set_name):
    """
    Generate atom input.
    """
    MAX_ATOMNUM = config['data']['max_atommun']
    ATOM_PAD_VAL = config['data']['atom_pad_val']
    REPLICA_NUM = config['augmentation']['replica_num']

    os.makedirs(f"{folder_path}/Trans/{REPLICA_NUM}/", exist_ok=True)

    # Peptide information
    id = df_enu['ID'].to_numpy()
    smiles = df_enu['SMILES'].to_numpy()
    y = df['permeability'].to_numpy()
    y = np.clip(y, config['data']['lower_limit'], config['data']['upper_limit']).repeat(REPLICA_NUM)

    # Atoms mask
    atoms_mask = get_atoms_mask(mols, MAX_ATOMNUM)
    # Atoms features (Node)
    atoms_features = calculate_atoms_features(mols, MAX_ATOMNUM, ATOM_PAD_VAL)
    np.savez_compressed(f"{folder_path}/Trans/{REPLICA_NUM}/node_{REPLICA_NUM}_{set_name}.npz",
                        id=id,
                        smiles=smiles,
                        y=y,
                        atoms_mask=atoms_mask,
                        atoms_features=atoms_features)

    # Bonds type (Bond)
    bond = calculate_bond_type_matrix(mols, MAX_ATOMNUM)
    np.savez_compressed(f"{folder_path}/Trans/{REPLICA_NUM}/bond_{REPLICA_NUM}_{set_name}.npz", bond=bond)

    # Graph distance matrix (Graph)
    graph = calculate_graph_distance_matrix(mols, MAX_ATOMNUM, ATOM_PAD_VAL)
    np.savez_compressed(f"{folder_path}/Trans/{REPLICA_NUM}/graph_{REPLICA_NUM}_{set_name}.npz", graph=graph)

    # 3D distance matrix (Conf)
    conf = calculate_conf_distance_matrix(mols, MAX_ATOMNUM, ATOM_PAD_VAL)
    # ensure conf has expected shape (n_mols, MAX_ATOMNUM, MAX_ATOMNUM)
    try:
        conf = np.asarray(conf)
        if conf.ndim != 3 or conf.shape[1] != MAX_ATOMNUM or conf.shape[2] != MAX_ATOMNUM:
            n_mols = len(mols)
            conf_safe = np.full((n_mols, MAX_ATOMNUM, MAX_ATOMNUM), ATOM_PAD_VAL, dtype=float)
            # if conf provides some valid entries, copy those into safe array where possible
            try:
                min_n = min(conf.shape[0], n_mols)
                min_r = min(conf.shape[1] if conf.ndim>1 else 0, MAX_ATOMNUM)
                min_c = min(conf.shape[2] if conf.ndim>2 else 0, MAX_ATOMNUM)
                if conf.ndim == 3 and min_n>0 and min_r>0 and min_c>0:
                    conf_safe[:min_n, :min_r, :min_c] = conf[:min_n, :min_r, :min_c]
            except Exception:
                pass
            conf = conf_safe
    except Exception:
        n_mols = len(mols)
        conf = np.full((n_mols, MAX_ATOMNUM, MAX_ATOMNUM), ATOM_PAD_VAL, dtype=float)

    np.savez_compressed(f"{folder_path}/Trans/{REPLICA_NUM}/conf_{REPLICA_NUM}_{set_name}.npz", conf=conf)


    # # NOTE: You can also generate input data for different augmentation times
    # # For example:
    # total_list = list(range(0, len(df_enu)))
    # for replica_num in [1, 5, 10, 20, 30, 40, 50]:
    #     # IMPORTANT
    #     select_list = [total_list[i:i+replica_num] for i in range(0, len(total_list), 60)]
    #     select_list = sum(select_list, [])
    #     np.savez_compressed(f"{folder_path}/Trans/{replica_num}/node_{replica_num}_{set_name}.npz"
    #                         id=id[select_list],
    #                         smiles=smiles[select_list],
    #                         y=y[select_list],
    #                         atoms_mask=atoms_mask[select_list],
    #                         atoms_features=atoms_features[select_list])
