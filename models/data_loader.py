"""Data loading and preprocessing for protein structures.

This module provides utilities for loading protein structures from various
formats (BinaryCIF, PDB, mmCIF) and converting them to tensors suitable
for the protein design model.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from torch.utils.data import Dataset

from utils import get_logger

logger = get_logger(__name__)

# Standard amino acid vocabulary (20 canonical amino acids)
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {idx: aa for aa, idx in AA_TO_IDX.items()}

# Backbone atom names in standard order
BACKBONE_ATOMS = ['N', 'CA', 'C', 'O']


@dataclass
class ProteinStructure:
    """Container for protein structure data.

    Attributes:
        name: Protein identifier (e.g., PDB ID).
        sequence: Amino acid sequence as a string.
        coords: Backbone coordinates with shape [Length, 4, 3].
                The 4 atoms are [N, CA, C, O] in Angstroms.
        mask: Validity mask with shape [Length]. 1 for valid residues,
              0 for missing/invalid residues.
        chain_id: Chain identifier (e.g., 'A', 'B').

    Example:
        >>> protein = ProteinStructure(
        ...     name='1ABC',
        ...     sequence='ACDEFGHIKLMNPQRSTVWY',
        ...     coords=np.random.randn(20, 4, 3),
        ...     mask=np.ones(20),
        ...     chain_id='A'
        ... )
    """
    name: str
    sequence: str
    coords: np.ndarray  # [Length, 4, 3]
    mask: np.ndarray    # [Length]
    chain_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate structure data after initialization."""
        seq_len = len(self.sequence)
        if self.coords.shape[0] != seq_len:
            raise ValueError(
                f"Sequence length ({seq_len}) doesn't match "
                f"coordinates length ({self.coords.shape[0]})"
            )
        if self.mask.shape[0] != seq_len:
            raise ValueError(
                f"Sequence length ({seq_len}) doesn't match "
                f"mask length ({self.mask.shape[0]})"
            )
        if self.coords.shape != (seq_len, 4, 3):
            raise ValueError(
                f"Coordinates should have shape [{seq_len}, 4, 3], "
                f"got {self.coords.shape}"
            )


def load_structure_from_gemmi(
    structure_path: Path,
    chain_id: Optional[str] = None
) -> ProteinStructure:
    """Load protein structure from BinaryCIF/CIF/PDB using gemmi.

    Args:
        structure_path: Path to structure file (.bcif, .cif, or .pdb).
        chain_id: Specific chain to extract. If None, uses the first chain.

    Returns:
        ProteinStructure object containing coordinates and sequence.

    Raises:
        ImportError: If gemmi is not installed.
        FileNotFoundError: If structure file doesn't exist.
        ValueError: If no valid residues found or chain not found.

    Example:
        >>> structure = load_structure_from_gemmi(
        ...     Path('structures/1abc.bcif'),
        ...     chain_id='A'
        ... )
        >>> print(structure.sequence)
        'ACDEFGH...'
    """
    try:
        import gemmi
    except ImportError:
        raise ImportError(
            "gemmi is required for loading BinaryCIF/CIF files. "
            "Install with: pip install gemmi"
        )

    structure_path = Path(structure_path)
    if not structure_path.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_path}")

    # Load structure based on file extension
    if structure_path.suffix in ['.bcif', '.cif']:
        structure = gemmi.read_structure(str(structure_path))
    elif structure_path.suffix == '.pdb':
        structure = gemmi.read_pdb(str(structure_path))
    else:
        raise ValueError(
            f"Unsupported file format: {structure_path.suffix}. "
            f"Supported formats: .bcif, .cif, .pdb"
        )

    # Get the first model
    if len(structure) == 0:
        raise ValueError(f"No models found in {structure_path}")
    model = structure[0]

    # Select chain
    if chain_id is None:
        if len(model) == 0:
            raise ValueError(f"No chains found in {structure_path}")
        chain = model[0]
        chain_id = chain.name
        logger.info(f"No chain specified, using first chain: {chain_id}")
    else:
        chain = model.find_chain(chain_id)
        if chain is None:
            available_chains = [c.name for c in model]
            raise ValueError(
                f"Chain '{chain_id}' not found in {structure_path}. "
                f"Available chains: {available_chains}"
            )

    # Extract residues
    residues = []
    coords_list = []
    mask_list = []

    for residue in chain:
        # Skip non-protein residues (water, ligands, etc.)
        if not residue.is_protein():
            continue

        # Get single-letter amino acid code
        aa = gemmi.find_tabulated_residue(residue.name).one_letter_code
        if aa not in AA_TO_IDX:
            logger.warning(
                f"Unknown amino acid '{aa}' ({residue.name}) in "
                f"{structure_path}, chain {chain_id}, residue {residue.seqid}. Skipping."
            )
            continue

        # Extract backbone atom coordinates
        atom_coords = np.zeros((4, 3), dtype=np.float32)
        atom_mask = np.zeros(4, dtype=np.float32)

        for atom_idx, atom_name in enumerate(BACKBONE_ATOMS):
            atom = residue.find_atom(atom_name, '*')  # '*' matches any altloc
            if atom:
                pos = atom.pos
                atom_coords[atom_idx] = [pos.x, pos.y, pos.z]
                atom_mask[atom_idx] = 1.0

        # Only include residue if all backbone atoms are present
        if np.all(atom_mask == 1.0):
            residues.append(aa)
            coords_list.append(atom_coords)
            mask_list.append(1.0)
        else:
            logger.warning(
                f"Missing backbone atoms in {structure_path}, "
                f"chain {chain_id}, residue {residue.seqid} ({aa}). Skipping."
            )

    if len(residues) == 0:
        raise ValueError(
            f"No valid residues found in {structure_path}, chain {chain_id}"
        )

    # Convert to arrays
    sequence = ''.join(residues)
    coords = np.stack(coords_list, axis=0)  # [Length, 4, 3]
    mask = np.array(mask_list, dtype=np.float32)  # [Length]

    protein = ProteinStructure(
        name=structure_path.stem,
        sequence=sequence,
        coords=coords,
        mask=mask,
        chain_id=chain_id
    )

    logger.info(
        f"Loaded {structure_path.name}, chain {chain_id}: "
        f"{len(sequence)} residues"
    )

    return protein


def load_structure_from_biopython(
    structure_path: Path,
    chain_id: Optional[str] = None
) -> ProteinStructure:
    """Load protein structure from PDB file using BioPython.

    Args:
        structure_path: Path to PDB file.
        chain_id: Specific chain to extract. If None, uses the first chain.

    Returns:
        ProteinStructure object containing coordinates and sequence.

    Raises:
        ImportError: If BioPython is not installed.
        FileNotFoundError: If structure file doesn't exist.
        ValueError: If no valid residues found or chain not found.

    Example:
        >>> structure = load_structure_from_biopython(
        ...     Path('structures/1abc.pdb'),
        ...     chain_id='A'
        ... )
    """
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.Polypeptide import protein_letters_3to1
    except ImportError:
        raise ImportError(
            "BioPython is required. Install with: pip install biopython"
        )

    structure_path = Path(structure_path)
    if not structure_path.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_path}")

    # Parse PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(structure_path.stem, str(structure_path))

    # Get first model
    model = structure[0]

    # Select chain
    if chain_id is None:
        chain = next(model.get_chains())
        chain_id = chain.id
        logger.info(f"No chain specified, using first chain: {chain_id}")
    else:
        if chain_id not in model:
            available_chains = [c.id for c in model.get_chains()]
            raise ValueError(
                f"Chain '{chain_id}' not found. Available: {available_chains}"
            )
        chain = model[chain_id]

    # Extract residues
    residues = []
    coords_list = []
    mask_list = []

    for residue in chain:
        # Skip hetero residues (water, ligands)
        if residue.id[0] != ' ':
            continue

        # Get amino acid code
        try:
            aa = protein_letters_3to1[residue.resname]
        except KeyError:
            logger.warning(f"Unknown residue: {residue.resname}")
            continue

        if aa not in AA_TO_IDX:
            continue

        # Extract backbone atoms
        atom_coords = np.zeros((4, 3), dtype=np.float32)
        atom_mask = np.zeros(4, dtype=np.float32)

        for atom_idx, atom_name in enumerate(BACKBONE_ATOMS):
            if atom_name in residue:
                atom = residue[atom_name]
                atom_coords[atom_idx] = atom.coord
                atom_mask[atom_idx] = 1.0

        # Only include if all backbone atoms present
        if np.all(atom_mask == 1.0):
            residues.append(aa)
            coords_list.append(atom_coords)
            mask_list.append(1.0)

    if len(residues) == 0:
        raise ValueError(f"No valid residues found in {structure_path}")

    sequence = ''.join(residues)
    coords = np.stack(coords_list, axis=0)
    mask = np.array(mask_list, dtype=np.float32)

    protein = ProteinStructure(
        name=structure_path.stem,
        sequence=sequence,
        coords=coords,
        mask=mask,
        chain_id=chain_id
    )

    logger.info(f"Loaded {structure_path.name}: {len(sequence)} residues")

    return protein


def sequence_to_tensor(sequence: str) -> torch.Tensor:
    """Convert amino acid sequence to integer tensor.

    Args:
        sequence: Amino acid sequence as a string (e.g., 'ACDEFGH').

    Returns:
        Integer tensor with shape [Length] containing amino acid indices.

    Raises:
        ValueError: If sequence contains unknown amino acids.

    Example:
        >>> seq_tensor = sequence_to_tensor('ACDE')
        >>> print(seq_tensor)  # tensor([0, 1, 2, 3])
    """
    indices = []
    for aa in sequence:
        if aa not in AA_TO_IDX:
            raise ValueError(f"Unknown amino acid: {aa}")
        indices.append(AA_TO_IDX[aa])

    return torch.tensor(indices, dtype=torch.long)


def tensor_to_sequence(tensor: torch.Tensor) -> str:
    """Convert integer tensor to amino acid sequence.

    Args:
        tensor: Integer tensor with shape [Length] containing amino acid indices.

    Returns:
        Amino acid sequence as a string.

    Example:
        >>> tensor = torch.tensor([0, 1, 2, 3])
        >>> seq = tensor_to_sequence(tensor)
        >>> print(seq)  # 'ACDE'
    """
    sequence = ''.join([IDX_TO_AA[idx.item()] for idx in tensor])
    return sequence


class ProteinDataset(Dataset):
    """PyTorch Dataset for protein structures.

    This dataset loads protein structures from files and provides batched
    access with proper padding and masking.

    Args:
        structure_paths: List of paths to structure files.
        max_length: Maximum sequence length. Longer sequences are truncated.
        chain_ids: Optional list of chain IDs (one per structure).
        use_gemmi: If True, use gemmi for loading (supports BinaryCIF).
                   If False, use BioPython (PDB only).

    Example:
        >>> paths = [Path('1abc.pdb'), Path('2def.pdb')]
        >>> dataset = ProteinDataset(paths, max_length=500)
        >>> protein = dataset[0]
        >>> print(protein['coords'].shape)  # [Length, 4, 3]
    """

    def __init__(
        self,
        structure_paths: List[Path],
        max_length: int = 500,
        chain_ids: Optional[List[str]] = None,
        use_gemmi: bool = True
    ):
        self.structure_paths = [Path(p) for p in structure_paths]
        self.max_length = max_length
        self.chain_ids = chain_ids
        self.use_gemmi = use_gemmi

        if chain_ids is not None and len(chain_ids) != len(structure_paths):
            raise ValueError(
                f"Number of chain_ids ({len(chain_ids)}) must match "
                f"number of structure_paths ({len(structure_paths)})"
            )

        logger.info(
            f"Initialized ProteinDataset with {len(structure_paths)} structures"
        )

    def __len__(self) -> int:
        return len(self.structure_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and return a protein structure.

        Args:
            idx: Index of the structure to load.

        Returns:
            Dictionary containing:
                - 'coords': Backbone coordinates [Length, 4, 3]
                - 'sequence': Amino acid indices [Length]
                - 'mask': Validity mask [Length]
                - 'name': Protein name (string)

        Shape:
            - coords: [Length, 4, 3]
            - sequence: [Length]
            - mask: [Length]
        """
        structure_path = self.structure_paths[idx]
        chain_id = self.chain_ids[idx] if self.chain_ids else None

        # Load structure
        if self.use_gemmi:
            protein = load_structure_from_gemmi(structure_path, chain_id)
        else:
            protein = load_structure_from_biopython(structure_path, chain_id)

        # Truncate if needed
        seq_len = len(protein.sequence)
        if seq_len > self.max_length:
            logger.warning(
                f"Truncating {protein.name} from {seq_len} to {self.max_length}"
            )
            protein.sequence = protein.sequence[:self.max_length]
            protein.coords = protein.coords[:self.max_length]
            protein.mask = protein.mask[:self.max_length]

        # Convert to tensors
        coords = torch.from_numpy(protein.coords).float()
        sequence = sequence_to_tensor(protein.sequence)
        mask = torch.from_numpy(protein.mask).float()

        return {
            'coords': coords,
            'sequence': sequence,
            'mask': mask,
            'name': protein.name
        }


def collate_protein_batch(
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """Collate function for batching proteins with padding.

    Args:
        batch: List of protein dictionaries from ProteinDataset.

    Returns:
        Batched dictionary with padded tensors:
            - 'coords': [Batch, MaxLength, 4, 3]
            - 'sequence': [Batch, MaxLength]
            - 'mask': [Batch, MaxLength]
            - 'lengths': [Batch]
            - 'names': List of protein names

    Shape:
        - coords: [Batch, MaxLength, 4, 3]
        - sequence: [Batch, MaxLength]
        - mask: [Batch, MaxLength]
        - lengths: [Batch]
    """
    batch_size = len(batch)
    lengths = [item['coords'].shape[0] for item in batch]
    max_length = max(lengths)

    # Initialize padded tensors
    coords_batch = torch.zeros(batch_size, max_length, 4, 3)
    sequence_batch = torch.zeros(batch_size, max_length, dtype=torch.long)
    mask_batch = torch.zeros(batch_size, max_length)
    names = []

    # Fill in data
    for i, item in enumerate(batch):
        length = lengths[i]
        coords_batch[i, :length] = item['coords']
        sequence_batch[i, :length] = item['sequence']
        mask_batch[i, :length] = item['mask']
        names.append(item['name'])

    return {
        'coords': coords_batch,
        'sequence': sequence_batch,
        'mask': mask_batch,
        'lengths': torch.tensor(lengths, dtype=torch.long),
        'names': names
    }
