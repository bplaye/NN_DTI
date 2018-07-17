from __future__ import print_function
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import rdkit.Chem.Descriptors as Descriptors
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import rdkit.Chem.EState as EState
import rdkit.Chem.rdPartialCharges as rdPartialCharges
import rdkit.Chem.rdChemReactions as rdRxns
import copy
import rdkit.Chem.MolSurf as MolSurf
import numpy as np
att_dtype = np.float32


class Graph():
  '''Describes an undirected graph class'''

  def __init__(self):
    self.nodes = []
    self.num_nodes = 0
    self.edges = []
    self.num_edges = 0
    self.N_features = 0
    return

  def clone(self):
    '''clone() method to trick Theano'''
    return copy.deepcopy(self)

  def dump_as_matrices(self):
    # Bad input handling
    if not self.nodes:
      raise(ValueError, 'Error generating tensor for graph with no nodes')
    # if not self.edges:
    #   raise(ValueError, 'Need at least one bond!')

    # get number of nodes
    N_nodes = len(self.nodes)
    # get size of vertex and edge attribute vectors
    F_a, F_b = sizeAttributeVectors(molecular_attributes=self.molecular_attributes)

    # initialize output
    X_a = np.zeros((N_nodes, F_a, 1), dtype=np.float32)
    A = np.zeros((N_nodes, N_nodes, 1), dtype=np.float32)
    Aaug = np.zeros((N_nodes, N_nodes, 1), dtype=np.float32)
    X_b = np.zeros((N_nodes, N_nodes, F_b, 1), dtype=np.float32)

    if len(self.edges) != 0:
      edgeAttributes = np.vstack([x.attributes for x in self.edges])
    else:
      edgeAttributes = np.zeros(F_b)
    nodeAttributes = np.vstack([x.attributes for x in self.nodes])

    for i, node in enumerate(self.nodes):
      X_a[i, :, 0] = nodeAttributes[i]
      Aaug[i, i, 0] = 1.0  # include self terms

    for e, edge in enumerate(self.edges):
      (i, j) = edge.connects
      A[i, j, 0] = 1.0
      A[j, i, 0] = 1.0
      Aaug[i, j, 0] = 1.0
      Aaug[j, i, 0] = 1.0

      # Keep track of extra special bond types - which are nothing more than
      # bias terms specific to the bond type because they are all one-hot encoded
      # X_b[i, :, 0] += edgeAttributes[e]
      # X_b[j, :, 0] += edgeAttributes[e]
      X_b[i, j, :, 0] += edgeAttributes[e]
      X_b[j, i, :, 0] += edgeAttributes[e]

    return (X_a, X_b, A, Aaug)


class Node():
  '''Describes an attributed node in an undirected graph'''

  def __init__(self, i=None, attributes=np.array([], dtype=att_dtype)):
    self.i = i
    self.attributes = attributes  # 1D array
    self.neighbors = []  # (atom index, bond index)
    return


class Edge():
  '''Describes an attributed edge in an undirected graph'''

  def __init__(self, connects=(), i=None, attributes=np.array([], dtype=att_dtype)):
    self.i = i
    self.attributes = attributes  # 1D array
    self.connects = connects  # (atom index, atom index)
    return


# -------------------------------------------------------------------------------
## @brief      convert a molecule from RDKit class to Graph class
##
## @param      rdmol                 The mol of RDKit class
## @param      molecular_attributes  boolean, add of not extra attribute to atom
##                                   attributes
##
## @return     the same mol, of class Graph
##
def molToGraph(rdmol, molecular_attributes=False):

  # Initialize
  graph = Graph()
  graph.molecular_attributes = molecular_attributes

  # Calculate atom-level molecule descriptors
  attributes = [[] for i in rdmol.GetAtoms()]
  if molecular_attributes:
    labels = []
    [attributes[i].append(x[0])
        for (i, x) in enumerate(rdMolDescriptors._CalcCrippenContribs(rdmol))]
    labels.append('Crippen contribution to logp')

    [attributes[i].append(x[1])
        for (i, x) in enumerate(rdMolDescriptors._CalcCrippenContribs(rdmol))]
    labels.append('Crippen contribution to mr')

    [attributes[i].append(x)
        for (i, x) in enumerate(rdMolDescriptors._CalcTPSAContribs(rdmol))]
    labels.append('TPSA contribution')

    [attributes[i].append(x)
        for (i, x) in enumerate(rdMolDescriptors._CalcLabuteASAContribs(rdmol)[0])]
    labels.append('Labute ASA contribution')

    [attributes[i].append(x)
        for (i, x) in enumerate(EState.EStateIndices(rdmol))]
    labels.append('EState Index')

    rdPartialCharges.ComputeGasteigerCharges(rdmol)
    [attributes[i].append(float(a.GetProp('_GasteigerCharge')))
        for (i, a) in enumerate(rdmol.GetAtoms())]
    labels.append('Gasteiger partial charge')

    # Gasteiger partial charges sometimes gives NaN
    for i in range(len(attributes)):
      if np.isnan(attributes[i][-1]) or np.isinf(attributes[i][-1]):
        attributes[i][-1] = 0.0

    [attributes[i].append(float(a.GetProp('_GasteigerHCharge')))
        for (i, a) in enumerate(rdmol.GetAtoms())]
    labels.append('Gasteiger hydrogen partial charge')

    # Gasteiger partial charges sometimes gives NaN
    for i in range(len(attributes)):
      if np.isnan(attributes[i][-1]) or np.isinf(attributes[i][-1]):
        attributes[i][-1] = 0.0

  # Add bonds
  for bond in rdmol.GetBonds():
    edge = Edge()
    edge.i = bond.GetIdx()
    edge.attributes = bondAttributes(bond)
    edge.connects = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    graph.edges.append(edge)
  # Add atoms
  for k, atom in enumerate(rdmol.GetAtoms()):
    node = Node()
    node.i = atom.GetIdx()
    node.attributes = atomAttributes(atom, extra_attributes=attributes[k])
    for neighbor in atom.GetNeighbors():
      node.neighbors.append((
          neighbor.GetIdx(),
          rdmol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetIdx()
      ))
    graph.nodes.append(node)
  # Add counts, for convenience
  graph.num_edges = len(graph.edges)
  graph.num_nodes = len(graph.nodes)
  return graph


def padGraphTensor(old_tensor, new_dsize, tensor_type):
  '''This function takes an input tensor of dsize x dsize x Nfeatures and pads
  up the first two dimensions to new_dsize with zeros as needed'''

  old_shape = old_tensor.shape
  if tensor_type == 'adjacency':
    new_tensor = np.zeros((new_dsize, new_dsize, old_shape[2]), dtype=np.float32)
    for i in range(old_shape[0]):
      for j in range(old_shape[1]):
        for k in range(old_shape[2]):
          new_tensor[i, j, k] = old_tensor[i, j, k]
  elif tensor_type == 'bonds':
    new_tensor = np.zeros((new_dsize, new_dsize, old_shape[2], old_shape[3]), dtype=np.float32)
    for i in range(old_shape[0]):
      for j in range(old_shape[1]):
        for k in range(old_shape[2]):
          for l in range(old_shape[3]):
            new_tensor[i, j, k, l] = old_tensor[i, j, k, l]
  elif tensor_type == 'atoms':
    new_tensor = np.zeros((new_dsize, old_shape[1], old_shape[2]), dtype=np.float32)
    for i in range(old_shape[0]):
      for j in range(old_shape[1]):
        for k in range(old_shape[2]):
          new_tensor[i, j, k] = old_tensor[i, j, k]

  return new_tensor


def bondAttributes(bond):
  '''Returns a numpy array of attributes for an RDKit bond
  From Neural FP defaults:
  The bond features were a concatenation of whether the bond type was single, double, triple,
  or aromatic, whether the bond was conjugated, and whether the bond was part of a ring.
  '''
  # Initialize
  attributes = []
  # Add bond type
  attributes += oneHotVector(bond.GetBondTypeAsDouble(), [1.0, 1.5, 2.0, 3.0])
  # Add if is aromatic
  attributes.append(bond.GetIsAromatic())
  # Add if bond is conjugated
  attributes.append(bond.GetIsConjugated())
  # Add if bond is part of ring
  attributes.append(bond.IsInRing())

  # NEED THIS FOR TENSOR REPRESENTATION - 1 IF THERE IS A BOND
  attributes.append(1)

  return np.array(attributes, dtype=att_dtype)


def atomAttributes(atom, extra_attributes=[]):
  '''Returns a numpy array of attributes for an RDKit atom
  From ECFP defaults:
  <IdentifierConfiguration>
        <Property Name="AtomicNumber" Value="1"/>
        <Property Name="HeavyNeighborCount" Value="1"/>
        <Property Name="HCount" Value="1"/>
        <Property Name="FormalCharge" Value="1"/>
        <Property Name="IsRingAtom" Value="1"/>
    </IdentifierConfiguration>
    '''
  # Initialize
  attributes = []
  # Add atomic number
  attributes += oneHotVector(atom.GetAtomicNum(), [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999])
  # Add heavy neighbor count
  attributes += oneHotVector(len(atom.GetNeighbors()), [0, 1, 2, 3, 4, 5])
  # Add hydrogen count
  attributes += oneHotVector(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
  # Add formal charge
  attributes.append(atom.GetFormalCharge())
  # Add boolean if in ring
  attributes.append(atom.IsInRing())
  # Add boolean if aromatic atom
  attributes.append(atom.GetIsAromatic())

  attributes += extra_attributes

  return np.array(attributes, dtype=att_dtype)


def oneHotVector(val, lst):
  '''Converts a value to a one-hot vector based on options in lst'''
  if val not in lst:
    val = lst[-1]  # all atoms other than B C N O F P S Cl Br I are consider the same
  return list(map(lambda x: x == val, lst))


def sizeAttributeVector(molecular_attributes=False):
  m = AllChem.MolFromSmiles('CC')
  g = molToGraph(m, molecular_attributes=molecular_attributes)
  a = g.nodes[0]
  b = g.edges[0]
  return len(a.attributes) + len(b.attributes)


def sizeAttributeVectors(molecular_attributes=False):
  m = AllChem.MolFromSmiles('CC')
  g = molToGraph(m, molecular_attributes=molecular_attributes)
  a = g.nodes[0]
  b = g.edges[0]
  return len(a.attributes), len(b.attributes)
