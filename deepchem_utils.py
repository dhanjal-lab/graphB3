


""" DeepChem utils functions including edge indices """

import numpy as np
from rdkit import Chem
import logging
import os
import torch



class ChemicalFeaturesFactory:
    """This is a singleton class for RDKit base features."""
    _instance = None

    @classmethod
    def get_instance(cls):
        try:
            from rdkit import RDConfig
            from rdkit.Chem import ChemicalFeatures
        except ModuleNotFoundError:
            raise ImportError("This class requires RDKit to be installed.")

        if not cls._instance:
            fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
            cls._instance = ChemicalFeatures.BuildFeatureFactory(fdefName)
        return cls._instance



def encode_with_one_hot(val, allowable_set, include_unknown_set=False):
  
    # Check if the input value is not in the allowable set when include_unknown_set is False
    if not include_unknown_set:
        if val not in allowable_set:
            logging.info("Input {0} not in allowable set {1}".format(val, allowable_set))
    
    # Initialize the length of the one-hot vector
    if include_unknown_set:
        one_hot_length = len(allowable_set) + 1
    else:
        one_hot_length = len(allowable_set)
    
    # Create the one-hot vector
    one_hot = [0.0] * one_hot_length
    
    try:
        # Try to find the index of the input value in the allowable set
        index = allowable_set.index(val)
        one_hot[index] = 1.0  # Set the corresponding index to 1.0
    except ValueError:
        # If value is not found and include_unknown_set is True, set the last index to 1.0
        if include_unknown_set:
            one_hot[-1] = 1.0
    
    return one_hot



def atom_type_one_hot(smile, include_unknown_set=True):
   
    # Define the allowable set of atom symbols
    allowable_set = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
    
    # Convert SMILES string to RDKit molecule object
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError("Invalid SMILES string provided")
    
    # Initialize list to store one-hot encoded vectors for atoms
    one_hot_molecule = []
    
    # Iterate through atoms in the molecule
    for atom in mol.GetAtoms():
        # Get atom symbol
        atom_symbol = atom.GetSymbol()
        
        # Encode atom symbol using one_hot_encode function
        atom_one_hot = encode_with_one_hot(atom_symbol, allowable_set, include_unknown_set)
        atom_one_hot = np.array(atom_one_hot)
        atom_one_hot = torch.tensor(atom_one_hot)
        # Append the one-hot encoded vector to the molecule representation
        one_hot_molecule.append(atom_one_hot)
        
    one_hot_molecule = torch.stack(one_hot_molecule)
    
    return one_hot_molecule



def get_atom_formal_charge(smile, include_unknown_set=False):
    
    allowable_set = [-2, -1, 0, 1, 2]

    mol = Chem.MolFromSmiles(smile)

    formal_charges = []

    for atom in mol.GetAtoms():
        formal_charge = float(atom.GetFormalCharge())
        formal_charge_one_hot = encode_with_one_hot(formal_charge, allowable_set, include_unknown_set)
        formal_charge_one_hot = np.array(formal_charge_one_hot)
        formal_charge_one_hot = torch.tensor(formal_charge_one_hot)
        formal_charges.append(formal_charge_one_hot)
    
    formal_charges = torch.stack(formal_charges)

    return formal_charges



def get_hybridization_one_hot(smile, include_unknown_set=False):

    allowable_set = ["SP", "SP2", "SP3"]
    
    mol = Chem.MolFromSmiles(smile)
    
    hybrids = []

    for atom in mol.GetAtoms():
        hybrid = str(atom.GetHybridization())
        hybrid_one_hot = encode_with_one_hot(hybrid, allowable_set, include_unknown_set)
        hybrid_one_hot = np.array(hybrid_one_hot)
        hybrid_one_hot = torch.tensor(hybrid_one_hot)
        hybrids.append(hybrid_one_hot)

    hybrids = torch.stack(hybrids)

    return hybrids



def construct_hydrogen_bonding_info(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string provided")

    factory = ChemicalFeaturesFactory.get_instance()
    feats = factory.GetFeaturesForMol(mol)
    hydrogen_bonding = []
    for f in feats:
        hydrogen_bonding.append((f.GetAtomIds()[0], f.GetFamily()))

    one_hot_info = []
    for atom in mol.GetAtoms():
        one_hot = [0.0, 0.0] # Initialize one-hot encoding for the current atom
        atom_idx = atom.GetIdx()
        for hydrogen_bonding_tuple in hydrogen_bonding:
            if hydrogen_bonding_tuple[0] == atom_idx:
                if hydrogen_bonding_tuple[1] == "Donor":
                    one_hot[0] = 1.0
                elif hydrogen_bonding_tuple[1] == "Acceptor":
                    one_hot[1] = 1.0
        one_hot = np.array(one_hot)
        one_hot = torch.tensor(one_hot)
        one_hot_info.append(one_hot)

    one_hot_info = torch.stack(one_hot_info)

    return one_hot_info



def get_atom_is_in_aromatic_one_hot(smile):
    
    mol = Chem.MolFromSmiles(smile)

    aromatics = []

    for atom in mol.GetAtoms():
        aromatic = [float(atom.GetIsAromatic())]
        aromatic = np.array(aromatic)
        aromatic = torch.tensor(aromatic)
        aromatics.append(aromatic)
    
    aromatics = torch.stack(aromatics)

    return aromatics



def get_atom_total_degree_one_hot(smile, include_unknown_set=True):

    mol = Chem.MolFromSmiles(smile)

    allowable_set = [1, 2, 3, 4, 5]

    degrees = []

    for atom in mol.GetAtoms():
        degree = float(atom.GetTotalDegree())
        degree_one_hot = encode_with_one_hot(degree, allowable_set, include_unknown_set)
        degree_one_hot = np.array(degree_one_hot)
        degree_one_hot = torch.tensor(degree_one_hot)
        degrees.append(degree_one_hot)

    degrees = torch.stack(degrees)

    return degrees



def get_atom_total_num_Hs_one_hot(smile, include_unknown_set=False):

    mol = Chem.MolFromSmiles(smile)

    allowable_set = [0, 1, 2, 3]

    num_Hs = []

    for atom in mol.GetAtoms():
        num_h = float(atom.GetTotalNumHs())
        num_h_one_hot = encode_with_one_hot(num_h, allowable_set, include_unknown_set)
        num_h_one_hot = np.array(num_h_one_hot)
        num_h_one_hot = torch.tensor(num_h_one_hot)
        num_Hs.append(num_h_one_hot)

    num_Hs = torch.stack(num_Hs)

    return num_Hs



def get_chirality_one_hot(smile):

    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError("Invalid SMILES string provided")

    chiral_info = []
    for atom in mol.GetAtoms():
        one_hot = [0.0, 0.0] # Initialize one-hot encoding for the current atom
        try:
            chiral_type = atom.GetProp('_CIPCode')
            if chiral_type == "R":
                one_hot[0] = 1.0
            elif chiral_type == "S":
                one_hot[1] = 1.0
        except:
            pass
        
        one_hot = np.array(one_hot)
        one_hot = torch.tensor(one_hot)
        chiral_info.append(one_hot)

    # Convert list of tensors to a single tensor
    chiral_info = torch.stack(chiral_info)

    return chiral_info



def smiles_to_edge_indices(smile):
    
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    edge_index = []

    src, dest = [], []
    for bond in mol.GetBonds():
        # add edge list considering a directed graph
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [start, end]
        dest += [end, start]

    src = np.array(src)
    src_tensor = torch.tensor(src)
    edge_index.append(src_tensor)

    dest = np.array(dest)
    dest_tensor = torch.tensor(dest)
    edge_index.append(dest_tensor)

    edge_index = torch.stack(edge_index)
    
    return edge_index


