from madftnn.dataset.sqlite_database.hamiltonian_database_qhnet import HamiltonianDatabase_qhnet

import lmdb
import numpy as np
import torch

#!/usr/bin/env python3

import numpy as np

from torch_geometric.transforms.radius_graph import RadiusGraph
import torch

from tqdm import tqdm
# from cal_initH import cal_initH

import bisect
import pickle
from torch.utils.data import Dataset
import lmdb
import glob
from .buildblock import *
import copy
from madftnn.dataset.utils_escflow import AOData, Onsite_3idx_Overlap_Integral, build_molecule, build_AO_index, get_all_conventions
from madftnn.dataset.matrix_transforms import pack_upper_triangle, unpack_upper_triangle, _matrix_transform_single, get_convention_dict, _cut_matrix_3d, _cut_matrix_3d_last

MD17_DATASETS = [
    "water",
    "ethanol",
    "malondialdehyde",
    "uracil",
    "salicylic_acid",
    "naphthalene",
    "aspirin",
]

QH9_DATASETS = [
    "qh9",
]

PUBCHEM_DATASETS = [
    "pubchem",
]

class HamiltonianDataset_qhnet_clean(torch.utils.data.Dataset):
    def __init__(
        self, 
        filepath, 
        enable_hami = True,
        dtype=torch.float32,
        remove_init=False,
        **kwargs
        ):
        super(HamiltonianDataset_qhnet_clean, self).__init__()
        
        self.dtype = dtype
        self.database = HamiltonianDatabase_qhnet(filepath)

        self.enable_hami = enable_hami
        self.remove_init=remove_init
        
    def __len__(self):
        return len(self.database)
    
    def pairwise_distances(self, coords):
        dists = np.zeros((coords.shape[0], coords.shape[0]))
        if len(coords.shape)!=2 or coords.shape[1]!=3:
            raise ValueError(f"sorry, the coords shape is {coords.shape}, please check")
        dists = np.sqrt(
                    np.sum((coords.reshape(-1,1,3)-coords.reshape(1,-1,3))**2,axis = -1)+1e-10
            )
        return dists

    def __getitem__(self, idx):
        R, E, F, diag, non_diag, diag_init, non_diag_init, diag_mask, non_diag_mask, A, G, L = \
            self.database[idx]
        
        local_orbitals = []
        local_orbitals_number = 0
        for z in L:
            local_orbitals.append(
                tuple((int(z), int(l)) for l in self.database.get_orbitals(z))
            )
            local_orbitals_number += sum(2 * l + 1 for _, l in local_orbitals[-1])
        if self.remove_init:
            diag = diag -diag_init
            non_diag = non_diag -non_diag_init
        out= {'pos': R, 
                'energy': E,
                'forces': F,
                # "full_hamiltonian": torch.cat([t.view(1, -1) for t in H], dim=1).squeeze(0),
                # "full_hamiltonian": torch.block_diag(*H),
                # "overlap_matrix": torch.block_diag(*S),
                # "core_hamiltonian": torch.block_diag(*C),
                'edge_index': A, 
                'labels': G,
                'atomic_numbers': L,
                'molecule_size':F.shape[0],
                # "orbitals": orbitals,
                # "mask": masks,#torch.block_diag(*mask),
                # "mask_l1": torch.block_diag(*mask_l1),
                # "init_hamiltonian": torch.cat([t.view(1, -1) for t in initH], dim=1).squeeze(0),
                # "init_hamiltonian": torch.block_diag(*initH),
                # "batch":idx
                }
        if self.enable_hami:
            dis = self.pairwise_distances(R)
            non_diag_mask_dis = np.triu(np.ones_like(dis, dtype=bool), k=1)
            non_diag_dis = dis[non_diag_mask_dis+non_diag_mask_dis.T]
            dis_mask = np.exp(-0.09378197*non_diag_dis*non_diag_dis-0.61498875*non_diag_dis-3.61389775)    #QH9 gap

            out.update({
                    'diag_hamiltonian': diag,
                        'non_diag_hamiltonian': non_diag,
                        'diag_mask': diag_mask,
                        'non_diag_mask': non_diag_mask,
                        "mask_l1": dis_mask})



        return out
    
DATA_SPLIT_RATIO = {'buckyball_catcher':[600./6102,50./6102, 1 - 650./6102],
            'double_walled_nanotube':[800./5032,100./5032, 1 - 900./5032],
            'AT_AT':[ 3000./19990, 200./19990, 1 - 3200./19990],
            'AT_AT_CG_CG':[ 2000./10153, 200./10153, 1 - 2200./10153],
            'stachyose':[ 8000./27138, 800./27138, 1 - 8800./27138],
            'DHA':[ 8000./69388, 800./69388, 1 - 8800./69388],
            'Ac_Ala3_NHMe':[ 6000./85109, 600./85109, 1 - 6600./85109],
        }
SYSTEM_REF = {
        "Ac_Ala3_NHMe":  
        -620662.75,
        "AT_AT": 
        -1154896.6,
        "AT_AT_CG_CG": 
        -2329950.2,
        "DHA":
        -631480.2,
        "stachyose":
        -1578839.0,
        "buckyball_catcher": # buckyball_catcher/radius3_broadcast_kmeans
        -2877475.2,
        "double_walled_nanotube": # double_walled_nanotube/radius3_broadcast_kmeans
        -7799787.0,
}


# when train ratio is -1, we can use this pre-defined split ratio
def get_data_default_config(data_name):
    # train ratio , val ratio,test ratio can be int or float.
    train_ratio,val_ratio,test_ratio = None,None,None
    if any(name in data_name.lower() for name in QH9_DATASETS):
        train_ratio,val_ratio,test_ratio = 0.8,0.1,0.1
        atom_reference = np.zeros([20])
        system_ref = 0.
    elif any(name in data_name.lower() for name in PUBCHEM_DATASETS):
        train_ratio,val_ratio,test_ratio = 0.8,0.1,0.1
        atom_reference = np.array([0.0000, -376.3395, 0.0000, 0.0000, 0.0000,
                        0.0000,-23905.9824,-34351.3164,-47201.4062,0.0000,
                        0.0000,0.0000,0.0000,0.0000,0.0000,
                        -214228.1250,-249841.3906])
        system_ref = 0.
    # mdi7 datasets
    elif any(name in data_name.lower() for name in MD17_DATASETS):
        atom_reference = np.zeros([100])
        system_ref = 0.
        train_ratio,val_ratio,test_ratio = 0.8,0.1,0.1
    else:
        atom_reference = np.zeros([20])
        system_ref = SYSTEM_REF[data_name]
        train_ratio,val_ratio,test_ratio = DATA_SPLIT_RATIO[data_name]
    return atom_reference,system_ref,train_ratio,val_ratio,test_ratio

def get_full_energy(data_name, energy, atomic_numbers):
    atom_reference, system_ref,_,_,_ = get_data_default_config(data_name)
    unique,counts = np.unique(atomic_numbers, return_counts=True)
    energy += np.sum(atom_reference[unique]*counts)
    energy += system_ref
    return energy
    

class LmdbDataset(Dataset):
    r"""Dataset class to load from LMDB files containing relaxation
    trajectories or single point computations.
    Useful for Structure to Energy & Force (S2EF), Initial State to
    Relaxed State (IS2RS), and Initial State to Relaxed Energy (IS2RE) tasks.
    Args:
            @path: the path to store the data
            @task: the task for the lmdb dataset
            @split: splitting of the data
            @transform: some transformation of the data
    """
    energy = 'energy'
    forces = 'forces'
    def __init__(self, path,
                 data_name = "pubchem",
                 transforms = [],
                 enable_hami = True,
                 old_blockbuild = False,
                 remove_init = False,
                 remove_atomref_energy = False,
                 Htoblock_otf = True, ## on save H matrix, H to block is process in collate unifined for memory saving.
                 basis = "def2-tzvp"):
        super(LmdbDataset, self).__init__()
        self.path = path
        self.data_name = data_name
        self.basis = basis
        if data_name.lower() == "pubchem":
            if basis != "def2-tzvp":
                raise ValueError("sorry, when using pubchem the basis should be def2-tzvp")
        self.atom_reference, self.system_ref, _,_,_ = get_data_default_config(data_name)
        db_paths = []
        if isinstance(path,str):
            if path.endswith("lmdb"):
                db_paths.append(path)
            else:
                db_paths.extend(glob.glob(path+"/*.lmdb"))
        elif isinstance(path,list):
            for p in path:
                if p.endswith("lmdb"):
                    db_paths.append(p)
                else:
                    db_paths.extend(glob.glob(p+"/*.lmdb"))
        # print(db_paths)
        assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"
        self.enable_hami = enable_hami
        self._keys, self.envs = [], []
        self.db_paths = sorted(db_paths)
        self.open_db()
        self.transforms = transforms
        self.remove_init = remove_init
        self.remove_atomref_energy = remove_atomref_energy
        self.conv, self.orbitals_ref, self.mask,self.chemical_symbols = None,None,None,None
        self.old_blockbuild = old_blockbuild
        self.Htoblock_otf = Htoblock_otf
        if self.enable_hami:
            if (not self.old_blockbuild):
                self.conv, _, self.mask,_ = get_conv_variable_lin(basis)
            else:
                self.conv, self.orbitals_ref, self.mask,self.chemical_symbols = get_conv_variable(basis)

        self.set_attr_escflow_datset()

    def set_attr_escflow_datset(self):
        #self.set_atoms()
        #self.Q_dict = Onsite_3idx_Overlap_Integral(atom_list=self.atom_list, basis=self.basis).Q_table()
        self.set_conventions()
        #self.setup_Q()

    def set_atoms(self):
        name = self.data_name.lower()
        if "water" in name:
            self.atoms = [8, 1, 1]
            self.atom_list = ["O", "H"]
            self.hamiltonian_size = 24
        elif "ethanol" in name:
            self.atoms = [6, 6, 8, 1, 1, 1, 1, 1, 1]
            self.atom_list = ["C", "O", "H"]
            self.hamiltonian_size = 72
        elif "malondialdehyde" in name:
            self.atoms = [6, 6, 6, 8, 8, 1, 1, 1, 1]
            self.atom_list = ["C", "O", "H"]
            self.hamiltonian_size = 90
        elif "uracil" in name:
            self.atoms = [6, 6, 7, 6, 7, 6, 8, 8, 1, 1, 1, 1]
            self.atom_list = ["C", "N", "O", "H"]
            self.hamiltonian_size = 132
        elif "aspirin" in name:
            self.atoms = [6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1]
            self.atom_list = ["C", "O", "H"]
            raise NotImplementedError
        else:
            raise NotImplementedError

    def set_orbitals(self):
        orbitals_ref = {}
        orbitals_ref[1] = np.array([0, 0, 1])  # H: 2s 1p
        orbitals_ref[6] = np.array([0, 0, 0, 1, 1, 2])  # C: 3s 2p 1d
        orbitals_ref[7] = np.array([0, 0, 0, 1, 1, 2])  # N: 3s 2p 1d
        orbitals_ref[8] = np.array([0, 0, 0, 1, 1, 2])  # O: 3s 2p 1d
        self.orbitals_ref = orbitals_ref

    @staticmethod
    def construct_orbital_l_index(AO_lm_index):
        idx = 0
        AO_l_index = []
        while True:
            if idx >= len(AO_lm_index):
                break
            AO_l_index.append(AO_lm_index[idx].item())
            idx += 2 * AO_lm_index[idx] + 1
        return torch.tensor(AO_l_index)

    def open_db(self):
        for db_path in self.db_paths:
            self.envs.append(self.connect_db(db_path))
            # Try to get length from the database
            length_bytes = self.envs[-1].begin().get("length".encode("ascii"))
            if length_bytes is not None:
                # Standard format: length is stored as a pickled value
                length = pickle.loads(length_bytes)
            else:
                # Fallback: use database statistics to get number of entries
                with self.envs[-1].begin() as txn:
                    length = txn.stat()['entries']
            self._keys.append(list(range(length)))
        keylens = [len(k) for k in self._keys]
        self._keylen_cumulative = np.cumsum(keylens).tolist()
        self.num_samples = sum(keylens)

    def __len__(self):
        return self.num_samples

    @staticmethod
    def unpack_upper_triangle(packed: np.ndarray, h_dim: int):
        return unpack_upper_triangle(packed, h_dim)

    def matrix_transform(self, hamiltonian, atoms):
        return _matrix_transform_single(hamiltonian, atoms, self.convention_dict[self.convention])

    def set_conventions(self):
        if any(name in self.data_name.lower() for name in ["water", "ethanol", "malondialdehyde", "uracil", "aspirin"]):
            self.convention_dict = get_convention_dict()
            self.convention = "pyscf_def2svp_to_e3nn"
        elif "qh9" in self.data_name.lower():
            self.convention_dict = get_convention_dict()
            self.convention = "pyscf_def2svp_to_e3nn"
        else:
            raise NotImplementedError

    def get_mol(self, data_dict, orb_energy_and_coeff=False):
        num_nodes = torch.tensor(data_dict["num_nodes"], dtype=torch.int64)
        atoms = torch.tensor(np.frombuffer(data_dict["atoms"], np.int32), dtype=torch.int64)
        pos = torch.tensor(np.frombuffer(data_dict["pos"], np.float64).reshape(-1, 3), dtype=torch.float64)
        dft_energy = torch.tensor(data_dict["dft_energy"], dtype=torch.float64)
        dft_forces = torch.tensor(np.frombuffer(data_dict["dft_forces"], np.float64).reshape(-1, 3), dtype=torch.float64) # unit: Eh/Bohr
        h_dim = data_dict["h_dim"] # sum of orbital dimensions
        packed_hamiltonian = np.frombuffer(data_dict["packed_hamiltonian"], np.float64)
        packed_overlap = np.frombuffer(data_dict["packed_overlap"], np.float64)
        packed_init_ham = np.frombuffer(data_dict["packed_initial_hamiltonian"], np.float64)
        orbital_energies = np.frombuffer(data_dict["orbital_energies"], np.float64)
        packed_orbital_coeff = np.frombuffer(data_dict["packed_orbital_coefficients"], np.float64)
        #packed_dm0 = np.frombuffer(data_dict["packed_dm0"], np.float64)
        
        hamiltonian = torch.from_numpy(self.unpack_upper_triangle(packed_hamiltonian, h_dim)).to(torch.float64)
        overlap_matrix = torch.from_numpy(self.unpack_upper_triangle(packed_overlap, h_dim)).to(torch.float64)
        initial_hamiltonian = torch.from_numpy(self.unpack_upper_triangle(packed_init_ham, h_dim)).to(torch.float64)
        orbital_coefficients = torch.from_numpy(self.unpack_upper_triangle(packed_orbital_coeff, h_dim)).to(torch.float64)

        
        hamiltonian = self.matrix_transform(hamiltonian, atoms)
        overlap_matrix = self.matrix_transform(overlap_matrix, atoms)
        initial_hamiltonian = self.matrix_transform(initial_hamiltonian, atoms)
        
        AO_index = build_AO_index(build_molecule(atoms, pos), "def2-svp")
        AO_l_index = self.construct_orbital_l_index(AO_index[1])
        
        edge_index = []
        for i in range(len(atoms)):
            for j in range(len(atoms)):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.int64).t().contiguous()
        full_edge_index = edge_index
        
        ret_data = AOData(
            pos=pos,
            atomic_numbers=atoms.view(-1, 1),
            forces=dft_forces,
            energy=dft_energy.view(1, 1),
            fock=hamiltonian.reshape(1, h_dim, h_dim),
            init_fock=initial_hamiltonian.reshape(1, h_dim, h_dim),
            overlap=overlap_matrix.reshape(1, h_dim, h_dim),
            orbital_energies=torch.from_numpy(orbital_energies.copy()).reshape(1, h_dim),
            orbital_coefficients=orbital_coefficients.reshape(1, h_dim, h_dim),
            
            #energy=energy.view(1, 1),
            #AO_index=AO_index,
            #AO_l_index=AO_l_index,
            #AO_l_index_len=torch.tensor(len(AO_l_index), dtype=torch.int64).view(1, 1),
            #num_atoms=num_nodes.view(1, 1),
            #Q=self.Q,
            #h_dim=torch.tensor(h_dim, dtype=torch.int64).view(1, 1),
            #full_edge_index=full_edge_index,
        )
        """
        ret_data = AOData(
            pos=pos,
            atoms=atoms.view(-1, 1),
            pyscf_energy=pyscf_energy.view(1, 1),
            dft_forces=dft_forces,
            hamiltonian=hamiltonian.reshape(1, h_dim, h_dim),
            overlap=overlap_matrix.reshape(1, h_dim, h_dim),
            init_ham=initial_hamiltonian.reshape(1, h_dim, h_dim),
            AO_index=AO_index,
            AO_l_index=AO_l_index,
            AO_l_index_len=torch.tensor(len(AO_l_index), dtype=torch.int64).view(1, 1),
            num_atoms=num_nodes.view(1, 1),
            Q=self.Q,
            h_dim=torch.tensor(h_dim, dtype=torch.int64).view(1, 1),
            full_edge_index=full_edge_index,
        )
        """

        return ret_data

    def __getitem__(self, idx):
        # Data(file_name='/data/recalculated_data/adequate-worm/GePar3-yl/111255333_pos.npy.npz',
        # atomic_numbers=[68, 1], energy=[1, 1], forces=[68, 3], 
        # init_fock=[1158, 1158], fock=[1158, 1158], num_nodes=68, pos=[68, 3],
        # edge_index=[2, 1388], labels=[68], num_labels=5, 
        # grouping_graph=[2, 1112], interaction_graph=[2, 214], id=0)

        # Figure out which db this should be indexed from.
        db_idx = bisect.bisect(self._keylen_cumulative, idx)
        # Extract index of element within that db.
        el_idx = idx
        if db_idx != 0:
            el_idx = idx - self._keylen_cumulative[db_idx - 1]
        assert el_idx >= 0

        if "escflow" in self.data_name:
            db_env = self.envs[db_idx]
            with db_env.begin() as txn:
                key = int(idx).to_bytes(length=4, byteorder="big")
                data_dict = txn.get(key)
            data_dict = pickle.loads(data_dict)
            data_object = self.get_mol(data_dict, orb_energy_and_coeff=True)
        else:

            # Return features.
            datapoint_pickled = (
                self.envs[db_idx]
                .begin()
                .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
            )
            data_object = pickle.loads(datapoint_pickled)
        data_object.id = el_idx #f"{db_idx}_{el_idx}"

 
        for transform in self.transforms:
            data_object = transform(data_object)
        out = {'pos': data_object.pos.numpy().astype(np.float32), 
            'forces': data_object.forces.numpy().astype(np.float32),
            # 'edge_index': data_object.edge_index.numpy(), 
            # 'labels': data_object.labels.numpy(),
            'atomic_numbers': data_object.atomic_numbers.numpy(),
            'molecule_size':data_object.pos.shape[0],
            "idx":idx
            }
        
        energy = data_object.energy.numpy()
        out["pyscf_energy"] = copy.deepcopy(energy.astype(np.float32))  # this is pyscf energy ground truth
        if self.remove_atomref_energy:
            unique,counts = np.unique(out["atomic_numbers"],return_counts=True)
            energy = energy - np.sum(self.atom_reference[unique]*counts)
            energy = energy - self.system_ref
            
        out["energy"] = energy.astype(np.float32) # this is used from model training, mean/ref is removed.
        
        if self.enable_hami:
            if self.remove_init:
                data_object.fock = data_object.fock - data_object.init_fock
            if self.Htoblock_otf == True:
                out.update({"buildblock_mask":self.mask,
                            "max_block_size":self.conv.max_block_size,
                            "fock":data_object.fock.numpy().astype(np.float32)
                            })
            else:
                diag,non_diag,diag_mask,non_diag_mask = None,None,None,None
                if (not self.old_blockbuild):
                    diag,non_diag,diag_mask,non_diag_mask = matrixtoblock_lin(data_object.fock.numpy().astype(np.float32),
                                                                            data_object.atomic_numbers.numpy(),
                                                                            self.mask,self.conv.max_block_size)
                else:
                    H = data_object.fock
                    initH = data_object.init_fock
                    Z = data_object.atomic_numbers

                    diag, non_diag, diag_init, non_diag_init, diag_mask, non_diag_mask = split2blocks(
                        matrix_transform(H,Z,self.conv).numpy(),
                        matrix_transform(initH,Z,self.conv).numpy(),
                        Z.numpy(), self.orbitals_ref, self.mask, self.conv.max_block_size)
                out.update({'diag_hamiltonian': diag,
                        'non_diag_hamiltonian': non_diag,
                        'diag_mask': diag_mask,
                        'non_diag_mask': non_diag_mask})
            out.update({"init_fock":data_object.init_fock.numpy().astype(np.float32)})
            out.update({"s1e":data_object.overlap.numpy().astype(np.float32)})
            if hasattr(data_object, 'orbital_energies'):
                out.update({"orbital_energy":data_object.orbital_energies.numpy().astype(np.float32)})
            if hasattr(data_object, 'orbital_coefficients'):
                out.update({"orbital_coefficients":data_object.orbital_coefficients.numpy().astype(np.float32)})

        return out
    
    # def get_raw_data(self, idx):
    #     out = self.__getitem__(idx)
    #     db_idx = bisect.bisect(self._keylen_cumulative, idx)
    #     # Extract index of element within that db.
    #     el_idx = idx
    #     if db_idx != 0:
    #         el_idx = idx - self._keylen_cumulative[db_idx - 1]
    #     assert el_idx >= 0

    #     # Return features.
    #     datapoint_pickled = (
    #         self.envs[db_idx]
    #         .begin()
    #         .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
    #     )
    #     data_object = pickle.loads(datapoint_pickled)
        
    #     if self.remove_init:
    #             data_object.fock = data_object.fock - data_object.init_fock
        
    #     data_object.id = el_idx #f"{db_idx}_{el_idx}"

    #     out.update({})
    
    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1024,
        )
        return env
 
    def close_db(self):
        if not self.path.is_file():
            for env in self.envs:
                env.close()
            self.envs = []
        else:
            self.env.close()
            self.env = None

    def setup_Q(self):
        Q_blocks = []
        for l in range(60):
            block_diag_components = [self.Q_dict[z][l] for z in self.atoms]
            Q_blocks.append(torch.block_diag(*block_diag_components))
        Q = torch.stack(Q_blocks)  # [60, h_dim, h_dim]
        Q = self.matrix_transform(Q, torch.tensor(self.atoms)).permute(1, 2, 0) #[h_dim, h_dim, 60]
        Q[:, :, 16:40] = (
            Q[:, :, 16:40]
            .reshape(self.hamiltonian_size, self.hamiltonian_size, -1, 3)[:, :, :, [1, 2, 0]]
            .reshape(self.hamiltonian_size, self.hamiltonian_size, 24)
        )
        self.Q = Q


def matrix_transform(hamiltonian, atoms, convention):
    """Transform matrices according to orbital convention using PyTorch.
    
    This function reorders and applies sign changes to orbital matrices based on
    different quantum chemistry software conventions. Different software packages
    use different orbital ordering and sign conventions.
    
    Example:
        Transform from 6-31G to def2-SVP convention:
        - 6-31G: p orbitals ordered as [px, py, pz] 
        - def2-SVP: p orbitals ordered as [py, pz, px]
        - This function handles the reordering and sign changes
    
    Args:
        hamiltonian (torch.Tensor): Input matrices to transform, shape (..., n_orb, n_orb).
        atoms (torch.Tensor): Atomic numbers for the molecule (e.g., [6, 1, 1, 1] for CH3).
        convention_rule (Namespace): Orbital convention to use:
            - 'pyscf_def2-tzvp': def2-TZVP basis set convention (p: [pz, px, py])
            - 'pyscf_631G': 6-31G basis set convention (p: [pz, px, py])
            - 'pyscf_def2svp': def2-SVP basis set convention (p: [py, pz, px])
            - 'back2pyscf': Convert back to PySCF native convention (p: [pz, px, py])
              * Use this when you have matrices from other software and need to
                convert them back to PySCF format for density matrix calculations
              * Same basis as def2-SVP but with PySCF's native p-orbital ordering
    
    Returns:
        torch.Tensor: Transformed matrices with reordered orbitals and applied sign changes.
    """
    conv = convention_rule
    
    # Get device from hamiltonian tensor
    device = hamiltonian.device
    dtype = hamiltonian.dtype
    
    orbitals = ""
    orbitals_order = []
    for a in atoms:
        offset = len(orbitals_order)
        orbitals += conv.atom_to_orbitals_map[a.item()]
        orbitals_order += [idx + offset for idx in conv.orbital_order_map[a.item()]]

    transform_indices = []
    transform_signs = []
    for orb in orbitals:
        offset = sum(map(len, transform_indices))
        map_idx = conv.orbital_idx_map[orb]
        map_sign = conv.orbital_sign_map[orb]
        # Convert to torch tensors directly on the correct device
        transform_indices.append(torch.tensor(map_idx, device=device, dtype=torch.long) + offset)
        transform_signs.append(torch.tensor(map_sign, device=device, dtype=dtype))

    # Reorder according to orbitals_order
    transform_indices = [transform_indices[idx] for idx in orbitals_order]
    transform_signs = [transform_signs[idx] for idx in orbitals_order]
    
    # Concatenate using torch.cat instead of np.concatenate
    transform_indices = torch.cat(transform_indices)
    transform_signs = torch.cat(transform_signs)

    # Apply transformation using torch indexing
    hamiltonian_new = hamiltonian[..., transform_indices, :]
    hamiltonian_new = hamiltonian_new[..., :, transform_indices]
    
    # Apply signs using torch operations
    hamiltonian_new = hamiltonian_new * transform_signs.unsqueeze(-1)
    hamiltonian_new = hamiltonian_new * transform_signs.unsqueeze(-2)

    return hamiltonian_new


import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from typing import Dict

import h5py
import numpy as np
import torch
# file system
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import Dataset

from madftnn.dataset.utils_escflow import ANG2BOHR, eV2HA
from ase.data import chemical_symbols


def _to_tensor(data) -> torch.Tensor:
    """Utility converting numpy arrays to float64 torch tensors on CPU."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().double()
    arr = np.asarray(data)
    dtype = torch.float64 if arr.dtype.kind == "f" else torch.int64
    return torch.from_numpy(arr).to(dtype)


class RMD17_DFT(Dataset):
    """
    In-memory loader for HDF5 MD17 data that mimics the behaviour of the existing
    LMDB-based `MD17_DFT_Shard` dataset.

    Each sample is returned as an `AOData` object with the same fields so that
    downstream code can use the two datasets interchangeably. Loads all splits
    (train, val, test) by default.

    Parameters
    ----------
    root:
        Root directory containing the dataset folder.
    name:
        Dataset name (e.g., "water", "aspirin").
    prefix:
        Prefix for the dataset folder name (default: "md-").
    postfix:
        Postfix for the dataset folder name (default: "").
    load_orbitals:
        If ``True`` also include orbital energies / coefficients.
    include_density:
        Attach ``density_matrix`` and ``h_core`` tensors (default ``False``).
    all_features:
        Mimic ``MD17_DFT_Shard(..., all_features=True)`` by also exposing
        ``energy``/``force`` (Hartree / Hartree·Bohr⁻¹) and additional metadata.
    """

    def __init__(
        self,
        root: str,
        name: str,
        prefix: str = "",
        postfix: str = "",
        load_orbitals: bool = False,
        include_density: bool = False,
        all_features: bool = False,

        enable_hami: bool = True,
        old_blockbuild: bool = False,
        basis: str = "def2svp",
        remove_atomref_energy: bool = True,
        remove_init: bool = True,
        Htoblock_otf: bool = True,
    ) -> None:
        super().__init__()

        self.name = name
        assert name in ["rmd-aspirin", "rmd-ethanol", "rmd-naphthalene", "rmd-salicylic_acid"], f"Dataset name {name} not supported"
        self.prefix = "" if prefix is None else str(prefix)
        self.postfix = "" if postfix is None else str(postfix)
        self.folder = root
        self.h5_file = os.path.join(self.folder, "data.h5")

        self.enable_hami = enable_hami
        self.remove_atomref_energy = remove_atomref_energy
        self.remove_init = remove_init
        self.Htoblock_otf = Htoblock_otf
        self.atom_reference, self.system_ref, _,_,_ = get_data_default_config(self.name.replace("rmd-", ""))

        self.conv, self.orbitals_ref, self.mask,self.chemical_symbols = None,None,None,None
        self.old_blockbuild = old_blockbuild
        self.Htoblock_otf = Htoblock_otf
        self.basis = basis
        if self.enable_hami:
            if (not self.old_blockbuild):
                self.conv, _, self.mask,_ = get_conv_variable_lin(basis)
            else:
                self.conv, self.orbitals_ref, self.mask,self.chemical_symbols = get_conv_variable(basis)

        if not os.path.exists(self.h5_file):
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_file}")

        self.load_orbitals = load_orbitals
        self.include_density = include_density or all_features
        self.all_features = all_features

        # Splits are randomly generated in the common/data_utils.py
        with h5py.File(self.h5_file, "r") as f:
            self.metadata = dict(f["metadata"].attrs.items())
            splits = f["splits"]
            train_indices = splits["train_indices"][:]
            val_indices = splits["val_indices"][:]
            test_indices = splits["test_indices"][:]
            # Currently these indices are not used
        # Combine all splits: train + val + test
        self.indices = train_indices.tolist() + val_indices.tolist() + test_indices.tolist()

        # Orbital reference information for different atomic numbers
        self.full_orbitals = 14
        self.orbital_mask = {}
        self.orbitals_ref = {}
        self.orbitals_ref[1] = np.array([0, 0, 1])  # H: 2s 1p
        self.orbitals_ref[6] = np.array([0, 0, 0, 1, 1, 2])  # C: 3s 2p 1d
        self.orbitals_ref[7] = np.array([0, 0, 0, 1, 1, 2])  # N: 3s 2p 1d
        self.orbitals_ref[8] = np.array([0, 0, 0, 1, 1, 2])  # O: 3s 2p 1d

        if self.name == "rmd-aspirin":
            self.atoms = [6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1]
            self.atom_list = ["C", "O", "H"]
            self.hamiltonian_size = 222
        elif self.name == "rmd-ethanol":
            self.atoms = [6, 6, 8, 1, 1, 1, 1, 1, 1]
            self.atom_list = ["C", "O", "H"]
            self.hamiltonian_size = 72
        elif self.name == "rmd-naphthalene":
            self.atoms = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1]
            self.atom_list = ["C", "H"]
            self.hamiltonian_size = 180
        elif self.name == "rmd-salicylic_acid":
            self.atoms = [6, 6, 6, 8, 6, 6, 6, 6, 8, 8, 1, 1, 1, 1, 1, 1]
            self.atom_list = ["C", "N", "O", "H"]
            self.hamiltonian_size = 170
        else:
            raise NotImplementedError(f"Dataset name {self.name} not supported")

        self._prepare_structure_constants()
        self._cache: Dict[int, AOData] = {}

    # ------------------------------------------------------------------ utils
    def _prepare_structure_constants(self) -> None:
        """Derive molecule-level helpers from the first sample."""
        with h5py.File(self.h5_file, "r") as f:
            sample = f[f"sample_{int(self.indices[0])}"]
            atomic_numbers = sample["atomic_numbers"][:].astype(int)
            self.h_dim = int(sample.attrs["nbasis"])

        self.atomic_numbers = atomic_numbers.tolist()
        self.atom_tensor = torch.tensor(self.atomic_numbers, dtype=torch.int64)
        self.basis = self.metadata.get("basis", "def2-svp")
        self.functional = self.metadata.get("functional", "pbe")
        self.convention_dict = get_convention_dict()

        orbitals = []
        for Z in self.atoms:
            orbitals.append(tuple((int(Z), int(l)) for l in self.orbitals_ref[Z]))
        self.orbitals = tuple(orbitals)

        # Pre-compute Q table (same logic as MD17_DFT_Shard.setup_Q)
        self.atom_symbols = sorted({chemical_symbols[int(z)] for z in self.atomic_numbers})
        onsite = Onsite_3idx_Overlap_Integral(atom_list=self.atom_symbols, basis=self.basis)
        self.Q_dict = onsite.Q_table()

        blocks = []
        for l in range(60):
            parts = [self.Q_dict[int(Z)][l] for Z in self.atomic_numbers]
            blocks.append(torch.block_diag(*parts))
        Q = torch.stack(blocks)  # [60, h_dim, h_dim]
        Q = self._matrix_transform(Q, self.atom_tensor, "pyscf_def2svp_to_e3nn").permute(1, 2, 0)
        if Q.shape[-1] >= 40:
            Q[:, :, 16:40] = (
                Q[:, :, 16:40]
                .reshape(self.h_dim, self.h_dim, -1, 3)[:, :, :, [1, 2, 0]]
                .reshape(self.h_dim, self.h_dim, 24)
            )
        self.Q = Q.double()

    def _matrix_transform(self, matrices: torch.Tensor, atoms: torch.Tensor, convention: str) -> torch.Tensor:
        return _matrix_transform_single(matrices, atoms, self.convention_dict[convention])

    # ----------------------------------------------------------------- dataset
    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> AOData:
        if idx >= len(self.indices):
            raise IndexError(idx)
        if idx in self._cache:
            return self._cache[idx]

        file_idx = int(self.indices[idx])
        with h5py.File(self.h5_file, "r") as f:
            sample = f[f"sample_{file_idx}"]
            aodata = self._build_sample(sample, dataset_idx=idx, file_idx=file_idx)
            self._cache[idx] = aodata
            return aodata

    # ----------------------------------------------------------------- helpers
    def _build_sample(self, sample: h5py.Group, *, dataset_idx: int, file_idx: int) -> AOData:
        pos = _to_tensor(sample["positions"][:])
        atoms = _to_tensor(sample["atomic_numbers"][:]).view(-1, 1)
        energy_ev = sample.attrs["energy_ev"]
        forces_ev = _to_tensor(sample["forces_ev_ang"][:])
        energy_ha = sample.attrs["energy_ha"]
        forces_ha_bohr = _to_tensor(sample["forces_ha_bohr"][:])
        energy_ref_ev = sample.attrs["energy_ref_ev"]
        forces_ref_ev = _to_tensor(sample["forces_ref_ev"][:]) # eV/Angstrom

        energy = torch.tensor([[energy_ev]], dtype=torch.float64)
        forces = forces_ev

        overlap = _to_tensor(sample["overlap"][:]).reshape(self.h_dim, self.h_dim)
        hamiltonian = _to_tensor(sample["hamiltonian"][:]).reshape(self.h_dim, self.h_dim)
        init_ham = _to_tensor(sample["initial_hamiltonian"][:]).reshape(self.h_dim, self.h_dim)

        overlap = self._matrix_transform(overlap, atoms, "pyscf_def2svp_to_e3nn")
        hamiltonian = self._matrix_transform(hamiltonian, atoms, "pyscf_def2svp_to_e3nn")
        init_ham = self._matrix_transform(init_ham, atoms, "pyscf_def2svp_to_e3nn")

        mol_spec = build_molecule(self.atomic_numbers, pos.detach().cpu().numpy())
        #AO_index = build_AO_index(mol_spec, self.basis)
        #AO_l_index = construct_orbital_l_index(AO_index[1])
        #AO_l_index_len = torch.tensor([[AO_l_index.numel()]], dtype=torch.int64)

        connections = [[i, j] for i in range(pos.shape[0]) for j in range(pos.shape[0]) if i != j]
        full_edge_index = (
            torch.tensor(connections, dtype=torch.int64).t().contiguous()
            if connections
            else torch.zeros((2, 0), dtype=torch.int64)
        )

        data_object = AOData(
            pos=pos,
            atomic_numbers=atoms,
            energy=energy,
            forces=forces,
            #energy_ref=energy_ref_ev,
            #forces_ref=forces_ref_ev,
            fock=hamiltonian.view(1, self.h_dim, self.h_dim),
            overlap=overlap.view(1, self.h_dim, self.h_dim),
            init_fock=init_ham.view(1, self.h_dim, self.h_dim),
            #AO_index=AO_index,
            #AO_l_index=AO_l_index,
            #AO_l_index_len=AO_l_index_len,
            #num_atoms=torch.tensor([[pos.shape[0]]], dtype=torch.int64),
            #Q=self.Q,
            #h_dim=torch.tensor([[self.h_dim]], dtype=torch.int64),
            #full_edge_index=full_edge_index,
        )


        if self.load_orbitals:
            data_object.orbital_energies = _to_tensor(sample["orbital_energies"][:]).view(1, -1)
            data_object.orbital_coefficients = _to_tensor(sample["orbital_coefficients"][:])

        out = {'pos': data_object.pos.numpy().astype(np.float32), 
            'forces': data_object.forces.numpy().astype(np.float32),
            # 'edge_index': data_object.edge_index.numpy(), 
            # 'labels': data_object.labels.numpy(),
            'atomic_numbers': data_object.atomic_numbers.numpy(),
            'molecule_size':data_object.pos.shape[0],
            "idx":dataset_idx
            }
        if isinstance(data_object.energy, torch.Tensor):
            energy = data_object.energy.detach().cpu().numpy()
        else:
            energy = data_object.energy
        out["pyscf_energy"] = copy.deepcopy(energy.astype(np.float32))  # this is pyscf energy ground truth
        if self.remove_atomref_energy:
            unique,counts = np.unique(out["atomic_numbers"],return_counts=True)
            energy = energy - np.sum(self.atom_reference[unique]*counts)
            energy = energy - self.system_ref
            
        out["energy"] = energy.astype(np.float32) # this is used from model training, mean/ref is removed.

        if self.enable_hami:
            if self.remove_init:
                data_object.fock = data_object.fock - data_object.init_fock
            if self.Htoblock_otf == True:
                out.update({"buildblock_mask":self.mask,
                            "max_block_size":self.conv.max_block_size,
                            "fock":data_object.fock.numpy().astype(np.float32)
                            })
            else:
                diag,non_diag,diag_mask,non_diag_mask = None,None,None,None
                if (not self.old_blockbuild):
                    diag,non_diag,diag_mask,non_diag_mask = matrixtoblock_lin(data_object.fock.numpy().astype(np.float32),
                                                                            data_object.atomic_numbers.numpy(),
                                                                            self.mask,self.conv.max_block_size)
                else:
                    H = data_object.fock
                    initH = data_object.init_fock
                    Z = data_object.atomic_numbers

                    diag, non_diag, diag_init, non_diag_init, diag_mask, non_diag_mask = split2blocks(
                        matrix_transform(H,Z,self.conv).numpy(),
                        matrix_transform(initH,Z,self.conv).numpy(),
                        Z.numpy(), self.orbitals_ref, self.mask, self.conv.max_block_size)
                out.update({'diag_hamiltonian': diag,
                        'non_diag_hamiltonian': non_diag,
                        'diag_mask': diag_mask,
                        'non_diag_mask': non_diag_mask})
            out.update({"init_fock":data_object.init_fock.numpy().astype(np.float32)})
            out.update({"s1e":data_object.overlap.numpy().astype(np.float32)})
            if hasattr(data_object, 'orbital_energies'):
                out.update({"orbital_energy":data_object.orbital_energies.numpy().astype(np.float32)})
            if hasattr(data_object, 'orbital_coefficients'):
                out.update({"orbital_coefficients":data_object.orbital_coefficients.numpy().astype(np.float32)})

        return out