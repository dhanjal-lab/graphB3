""" Import Dataset """

import sys
import os
import numpy as np
import pandas as pd
import argparse
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, global_add_pool as gap
from torch_geometric.data import Data, Batch
from rdkit import Chem
from deepchem_utils import atom_type_one_hot, get_hybridization_one_hot, construct_hydrogen_bonding_info, get_atom_formal_charge
from deepchem_utils import get_atom_is_in_aromatic_one_hot, get_atom_total_degree_one_hot, get_atom_total_num_Hs_one_hot, get_chirality_one_hot
from deepchem_utils import smiles_to_edge_indices

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


""" Import Dataset """

# Argument parser
parser = argparse.ArgumentParser(description="Run GraphB3 test with configurable input files.")
parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file")
parser.add_argument("--model_weights", type=str, required=True, help="Path to the model weights file")
parser.add_argument("--results_csv", type=str, required=True, help="Path to save the results CSV file")
parser.add_argument("--output_dir", type=str, required=True, help="Parent directory to save all outputs")
args = parser.parse_args()

# Creating directories 
os.makedirs(args.output_dir, exist_ok=True)
barplots_dir = os.path.join(args.output_dir, "GraphB3_figures")
os.makedirs(barplots_dir, exist_ok=True)

try:
    test_molecules = pd.read_csv(args.input_csv, names=['SMILES'])
except FileNotFoundError:
    print(f"Error: Input CSV file '{args.input_csv}' not found.")
    exit(1)
except Exception as e:
    print(f"Error reading input CSV file: {e}")
    exit(1)

# Check for invalid SMILES
invalid_smiles = []
for index, row in test_molecules.iterrows():
    try:
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol is None:
            invalid_smiles.append(index)
        elif mol.GetNumAtoms() == 1:  # Check if it's a single atom
            print(f"Warning: SMILES at index {index} represents a single atom, not a compound.")
            invalid_smiles.append(index)
    except ValueError as e:
        print(f"Error processing SMILES at index {index}: {e}")
        invalid_smiles.append(index)
    except Exception:
        invalid_smiles.append(index)

if len(invalid_smiles) == len(test_molecules):
    print("Error: All SMILES in the input file are invalid.")
    exit(1)

if invalid_smiles:
    print(f"Warning: Found {len(invalid_smiles)} invalid SMILES in the input file.")
    print("Removing invalid SMILES from the dataset.")
    test_molecules = test_molecules.drop(invalid_smiles).reset_index(drop=True)

""" Get DeepChem Features """

class MolGraphConvFeaturizerWoLabels():

    def __init__(self):
        pass
    
    def check_valid_smiles(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            raise ValueError("Invalid SMILES string provided")
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 1:
            raise ValueError("The provided SMILES represents a single atom, not a compound.")

    def featurize_one(self, smile):
        
        self.check_valid_smiles(smile)
        
        encoder = MolGraphConvFeaturizerWoLabels()  
        atom_type = atom_type_one_hot(smile)
        formal_charge = get_atom_formal_charge(smile)
        hybridization = get_hybridization_one_hot(smile)
        hydrogen_bonding_info = construct_hydrogen_bonding_info(smile)
        aromatic = get_atom_is_in_aromatic_one_hot(smile)
        degree = get_atom_total_degree_one_hot(smile)
        num_Hs = get_atom_total_num_Hs_one_hot(smile)
        chirality = get_chirality_one_hot(smile)

        # Concatenate features
        node_features = torch.cat((atom_type, formal_charge, hybridization, hydrogen_bonding_info, 
                                           aromatic, degree, num_Hs, chirality), dim=1)
        
        return node_features 
    
    def get_edge_index(self, smile):

        encoder = MolGraphConvFeaturizerWoLabels()
        edge_index = smiles_to_edge_indices(smile)

        return edge_index
    
    def create_data_object(self, smile):
        node_features = self.featurize_one(smile)
        edge_index = self.get_edge_index(smile)
        data = Data(x=node_features, edge_index=edge_index)
        return data
    
    def get_graphs(self, smiles_series):
        
        smiles_list = smiles_series.tolist()
        
        graphs = []

        for smile in smiles_list:
            data_object = self.create_data_object(smile)
            graphs.append(data_object)
        
        graph_object = Batch.from_data_list(graphs)

        return graph_object
    
featurizerwolabels = MolGraphConvFeaturizerWoLabels()
mol_graphs = featurizerwolabels.get_graphs(test_molecules['SMILES'])

mol_graphs.x = mol_graphs.x.to(torch.float32).cuda()
mol_graphs.edge_index = mol_graphs.edge_index.cuda()

""" Prediction Model """

class GraphConvModel(torch.nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(GraphConvModel, self).__init__()
        torch.manual_seed(565)

        self.initial_conv = GraphConv(mol_graphs.num_features, embedding_size)
        self.conv1 = GraphConv(embedding_size, embedding_size)
        self.conv2 = GraphConv(embedding_size, embedding_size)
        self.conv3 = GraphConv(embedding_size, embedding_size)
        
        self.out = torch.nn.Linear(embedding_size, 1)  
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index, batch_index):
        hidden = self.initial_conv(x, edge_index)
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)

        hidden = self.conv1(hidden, edge_index)
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)

        hidden = self.conv2(hidden, edge_index)
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)

        hidden = self.conv3(hidden, edge_index)
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)

        hidden = gap(hidden, batch_index)
        
        out = self.out(hidden)
        return out

model = GraphConvModel(embedding_size=128, num_classes=2).to(device)
print(model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))


state_dict = torch.load(args.model_weights, map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=True)


test_loader = DataLoader(mol_graphs, batch_size=1, shuffle=False)

y_prob = []
y_pred = []
threshold = 0.0

model.eval()
with torch.no_grad():
    for batch in test_loader:
        outputs = model(batch.x.float(), batch.edge_index.long(), batch.batch.long())
        predicted_labels = (outputs >= threshold).view(-1).int()
        predicted_probs = torch.sigmoid(outputs)
        y_pred += predicted_labels.tolist()
        y_prob += predicted_probs.view(-1).tolist()

# Create a DataFrame with the real and predicted labels
test_results = pd.DataFrame({'y_pred': y_pred, 'y_prob': y_prob})


def label_mols(x):
    if x == 1:
        return 'BBB+'
    else:
        return 'BBB-'
    
test_results['status'] = test_results['y_pred'].apply(label_mols)

test_results_df = pd.concat([test_molecules['SMILES'], test_results], axis=1)

print(test_results_df)


""" Explainer Model """

from torch_geometric.explain import Explainer, GNNExplainer, unfaithfulness
import random

# explainer


explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(lr=0.0005, epochs=4000),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='binary_classification',
        task_level='graph',
        return_type='raw',
    ),
)


# Function to set the seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 454
all_explanations = []
unfaith_metrics = []

with torch.no_grad():
    with torch.set_grad_enabled(True):
        for data in test_loader:
            # Set the seed for reproducibility in each iteration
            set_seed(seed)
            
            # Extract the graph attributes from the DataLoader
            X = data.x.float()
            Edge_index = data.edge_index.long()
            Batch_index = data.batch.long()
        
            # Apply the explainer to get the explanations for the current graph
            explanation = explainer(x=X, edge_index=Edge_index, batch_index=Batch_index)
            metric = unfaithfulness(explainer, explanation)
        
            # Append explanations to the list
            all_explanations.append(explanation)
            unfaith_metrics.append(metric)

        test_explain_all = Batch.from_data_list(all_explanations)


test_results_df['Unfaith'] = unfaith_metrics

""" save the results """

results_path = os.path.join(args.output_dir, "bbb_example_results.csv")
test_results_df.to_csv(results_path)


""" get substructures from explainer results """

test_loader_exp = DataLoader(test_explain_all, batch_size=1, shuffle=False)

edge_masks_model = []

for i in test_loader_exp:
    edge_score = i.edge_mask
    edge_score = edge_score.t()
    edge_masks_model.append(edge_score)

edge_indices_model = []

for i in test_loader_exp:
    edge_idx = i.edge_index
    edge_idx = edge_idx.t()
    edge_indices_model.append(edge_idx)

edge_indices_cpu_model = [tensor.to('cpu') for tensor in edge_indices_model]
edge_masks_cpu_model = [tensor.to('cpu') for tensor in edge_masks_model]

test_exp_dict = {
    'edge_indices': edge_indices_cpu_model,
    'edge_masks': edge_masks_cpu_model
}

def average_bidirectional_edges(graphs_dict):
    result = {'edge_indices': [], 'edge_masks': []}
    
    for edges, masks in zip(graphs_dict['edge_indices'], graphs_dict['edge_masks']):
        averaged_edges = []
        averaged_masks = []
        
        for i in range(0, edges.size(0), 2):
            # Take the first edge of the bidirectional pair
            edge = edges[i].tolist()
            # Compute the average score of the bidirectional pair
            avg_mask = (masks[i].item() + masks[i+1].item()) / 2
            
            averaged_edges.append(edge)
            averaged_masks.append(avg_mask)
        
        result['edge_indices'].append(torch.tensor(averaged_edges))
        result['edge_masks'].append(torch.tensor(averaged_masks))
    
    return result

test_exp_dict_avg = average_bidirectional_edges(test_exp_dict)

def get_atom_indices(index, dict):
    print(f"Index: {index}, Length of edge_indices: {len(dict['edge_indices'])}, Length of edge_masks: {len(dict['edge_masks'])}")
    
    test_dict = {}
    for key, value in zip(dict['edge_indices'][index], dict['edge_masks'][index]):
        test_dict[tuple(key.tolist())] = value.item()

    window_dict = {f'{i/10}-{(i+1)/10}': [] for i in range(10)}

    for key, value in test_dict.items():
        window_key = f'{int(value*10)/10}-{int(value*10 + 1)/10}'
        unique_numbers = sorted(set(sum([key], ())))
        window_dict[window_key].extend(unique_numbers)

    for k in window_dict:
        window_dict[k] = sorted(set(window_dict[k]))

    return window_dict

index_list = test_results_df.index.tolist()

atom_indices = {index: get_atom_indices(index, test_exp_dict_avg) for index in index_list}

def generate_gradient_colors(start_color, end_color, n_colors):
    """
    Generate a list of gradient colors between two colors with reversed order,
    so higher indices get darker colors.

    Parameters:
    start_color (list): RGB values for the start color (light color).
    end_color (list): RGB values for the end color (dark color).
    n_colors (int): Number of colors to generate.

    Returns:
    list: List of RGB colors.
    """
    start_color = np.array(start_color)
    end_color = np.array(end_color)
    colors = [tuple((start_color + (end_color - start_color) * (i / (n_colors - 1))).tolist()) for i in range(n_colors)]
    return colors

from rdkit import Chem
from rdkit.Chem import Draw

def highlight_molecule(smiles, importance_dict, label, image_size=(500, 500), output_filename='nchem_highlight.png'):
    """
    Highlights a molecule with different colors based on the importance scores.

    Parameters:
    smiles (str): SMILES string of the molecule.
    importance_dict (dict): Dictionary with importance scores and atom indices.
    label (int): Label of the molecule (1 for default color, 0 for light blue).
    image_size (tuple): Size of the image (width, height).
    output_filename (str): Name of the output image file.

    Returns:
    None
    """
    # Create molecule from SMILES
    m = Chem.MolFromSmiles(smiles)

    # Define colors
    light_red = [1.0, 1.0, 1.0]  # Light red
    dark_red = [1.0, 0.1, 0.1]   # Dark red

    # Generate gradient colors
    label_1_colors = generate_gradient_colors(light_red, dark_red, 10)

    # Define colors
    light_blue = [1.0, 1.0, 1.0]  # Light blue
    dark_blue = [0.2, 0.4, 0.9]  # Dark blue

    # Generate gradient colors
    label_0_colors = generate_gradient_colors(light_blue, dark_blue, 10)

    # Define color schemes
    if label == 1:
        gradient_colors = label_1_colors
    else:
        gradient_colors = label_0_colors
    
    # Set sssAtoms property to None initially
    m.__sssAtoms = None

    # Create a drawer object
    drawer = Draw.MolDraw2DCairo(image_size[0], image_size[1])
    
    # Prepare highlight colors
    highlight_colors = {}
    for i, (score_range, atom_indices) in enumerate(importance_dict.items()):
        if atom_indices:  # only assign color if there are atoms to highlight
            color = gradient_colors[i]  # Get the color for the current score range
            for atom_index in atom_indices:
                highlight_colors[atom_index] = color
    
    # Set the sssAtoms property with the atom indices to be highlighted
    m.__sssAtoms = list(highlight_colors.keys())
    
    # Draw molecule with highlight
    opts = drawer.drawOptions()
    drawer.DrawMolecule(m, highlightAtoms=list(highlight_colors.keys()), highlightAtomColors=highlight_colors)
    
    drawer.FinishDrawing()
    drawer.WriteDrawingText(output_filename)

# Create output directory
output_dir = os.path.join(args.output_dir, "GraphB3_figures")
os.makedirs(output_dir, exist_ok=True)

# Iterate over the DataFrame and generate images
for idx in test_results_df.index:
    smiles = test_results_df.loc[idx, 'SMILES']
    label = test_results_df.loc[idx, 'y_pred']
    importance_dict = get_atom_indices(idx, test_exp_dict_avg)  # Ensure this returns a dict with the expected structure
    
    if importance_dict:  # Only generate the image if importance_dict is not empty
        output_filename = os.path.join(output_dir, f'test_chem_{idx}.png')
        highlight_molecule(smiles, importance_dict, label, output_filename=output_filename)










