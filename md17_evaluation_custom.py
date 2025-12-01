import os
import glob
import torch
from pyscf import gto, scf, dft
import time
from escflow_eval_utils import init_pyscf_mf, calc_dm0_from_ham, matrix_transform_single
from escflow_eval_utils import BOHR2ANG
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import warnings
import json

# Suppress FutureWarning about torch.load
warnings.filterwarnings('ignore', category=FutureWarning)

def process_single_molecule(pred_file_path, gt_file_path,
    unit="ang", xc="pbe0", basis="def2svp"
):
    dir_path = os.path.dirname(pred_file_path)
    calc_path = pred_file_path.replace("pred_", "calc_")

    pred_data = torch.load(pred_file_path)
    gt_data = torch.load(gt_file_path)

    data_index = gt_data["idx"]

    atoms = gt_data["atoms"]
    pos = gt_data["pos"] * BOHR2ANG

    calc_mf = init_pyscf_mf(atoms, pos, unit=unit, xc=xc, basis=basis)
    grad_frame = calc_mf.nuc_grad_method()
    NEW_CALC = False
    try:
        # Check if calculated data exists
        if os.path.exists(calc_path):
            calc_data = torch.load(calc_path)
            calc_energy = calc_data["calc_energy"]
            calc_forces = calc_data["calc_forces"]
            calc_mo_energy = calc_data["calc_mo_energy"]
            calc_mo_coeff = calc_data["calc_mo_coeff"]
            calc_ham = calc_data["hamiltonian"]
            calc_overlap = calc_data["overlap"]
            
        else:
            calc_data = gt_data.copy()  # Use copy to avoid modifying original
            start_time = time.time()
            calc_mf.kernel()
            calc_data["calc_time"] = time.time() - start_time
            calc_data["hamiltonian"] = torch.tensor(calc_mf.get_fock(dm=calc_mf.make_rdm1()), dtype=torch.float64)
            calc_data["overlap"] = torch.tensor(calc_mf.get_ovlp(), dtype=torch.float64)
            calc_data["density_matrix"] = torch.tensor(calc_mf.make_rdm1(), dtype=torch.float64)
            calc_data["method"] = "RKS"
            calc_data["xc"] = xc
            calc_data["basis"] = basis
            calc_data["scf_cycles"] = calc_mf.cycles
            calc_data["forces"] = torch.tensor(-grad_frame.kernel(), dtype=torch.float64)

            calc_overlap = calc_data["overlap"].unsqueeze(0) # (gt_overlap - calc_overlap) has float32 precision error (1e^-7)
            calc_ham = calc_data["hamiltonian"].unsqueeze(0)
            
            calc_density, calc_res = calc_dm0_from_ham(atoms, calc_overlap, calc_ham, transform=False)
            calc_energy = calc_mf.energy_tot(calc_density)
            calc_data["calc_energy"] = calc_energy

            calc_mo_energy = calc_res["orbital_energies"].squeeze().numpy()
            calc_mo_coeff = calc_res["orbital_coefficients"].squeeze().numpy()
            calc_data["calc_mo_energy"] = calc_mo_energy
            calc_data["calc_mo_coeff"] = calc_mo_coeff

            mo_occ = calc_mf.get_occ(calc_mo_energy, calc_mo_coeff)
            calc_data["mo_occ"] = mo_occ
            calc_forces = -grad_frame.kernel(mo_energy=calc_mo_energy, mo_coeff=calc_mo_coeff, mo_occ=mo_occ)
            calc_data["calc_forces"] = calc_forces

            # save calc_data
            torch.save(calc_data, calc_path)

            NEW_CALC = True

        if "calc_force" in pred_data and not NEW_CALC:
            pred_energy = pred_data["calc_energy"]
            pred_forces = pred_data["calc_forces"]
            pred_mo_energy = pred_data["calc_mo_energy"]
            pred_mo_coeff = pred_data["calc_mo_coeff"]
        else:
            calc_overlap = calc_data["overlap"].unsqueeze(0) # (gt_overlap - calc_overlap) has float32 precision error (1e^-7)
            pred_ham = matrix_transform_single(pred_data["pred_hamiltonian"].unsqueeze(0), atoms, convention="back2pyscf")
            
            pred_density, pred_res = calc_dm0_from_ham(atoms, calc_overlap, pred_ham, transform=False)
            pred_energy = calc_mf.energy_tot(pred_density)
            pred_data["calc_energy"] = pred_energy
            
            pred_mo_energy = pred_res["orbital_energies"].squeeze().numpy()
            pred_mo_coeff = pred_res["orbital_coefficients"].squeeze().numpy()
            pred_data["calc_mo_energy"] = pred_mo_energy
            pred_data["calc_mo_coeff"] = pred_mo_coeff
            
            mo_occ = calc_mf.get_occ(pred_mo_energy, pred_mo_coeff)
            pred_data["mo_occ"] = mo_occ
            pred_forces = -grad_frame.kernel(mo_energy=pred_mo_energy, mo_coeff=-pred_mo_coeff, mo_occ=mo_occ)
            pred_data["calc_forces"] = pred_forces

            # save pred_data
            torch.save(pred_data, pred_file_path)
        if "calc_force" in gt_data and not NEW_CALC:
            gt_energy = gt_data["calc_energy"]
            gt_forces = gt_data["calc_forces"]
            gt_mo_energy = gt_data["calc_mo_energy"]
            gt_mo_coeff = gt_data["calc_mo_coeff"]
        else:
            calc_overlap = calc_data["overlap"].unsqueeze(0) # (gt_overlap - calc_overlap) has float32 precision error (1e^-7)
            gt_ham = matrix_transform_single(gt_data["hamiltonian"].unsqueeze(0), atoms, convention="back2pyscf")
            
            gt_density, gt_res = calc_dm0_from_ham(atoms, calc_overlap, gt_ham, transform=False)
            gt_energy = calc_mf.energy_tot(gt_density)
            gt_data["calc_energy"] = gt_energy

            gt_mo_energy = gt_res["orbital_energies"].squeeze().numpy()
            gt_mo_coeff = gt_res["orbital_coefficients"].squeeze().numpy()
            gt_data["calc_mo_energy"] = gt_mo_energy
            gt_data["calc_mo_coeff"] = gt_mo_coeff

            mo_occ = calc_mf.get_occ(gt_mo_energy, gt_mo_coeff)
            gt_data["mo_occ"] = mo_occ
            gt_forces = -grad_frame.kernel(mo_energy=gt_mo_energy, mo_coeff=-gt_mo_coeff, mo_occ=mo_occ)
            gt_data["calc_forces"] = gt_forces
            
            # save gt_data
            torch.save(gt_data, gt_file_path)

        pred_forces_norm = np.linalg.norm(pred_forces, axis=1)
        calc_forces_norm = np.linalg.norm(calc_forces, axis=1)
        gt_forces_norm = np.linalg.norm(gt_forces, axis=1)

        num_occ = int(gt_data["atoms"].sum() / 2)

        # Extract occupied orbital energies only
        pred_mo_energy_occ = pred_mo_energy[:num_occ]
        gt_mo_energy_occ = gt_mo_energy[:num_occ]
        calc_mo_energy_occ = calc_mo_energy[:num_occ]


        result = {
            "data_index": data_index,

            "hamiltonian_diff (pred-gt)": abs(pred_ham - gt_ham).mean(),
            "hamiltonian_diff (pred-calc)": abs(pred_ham - calc_ham).mean(),
            "hamiltonian_diff (gt-calc)": abs(gt_ham - calc_ham).mean(),

            "pred_energy": pred_energy,
            "gt_energy": gt_energy,
            "calc_energy": calc_energy,
            
            "energy_diff (pred-gt)": abs(pred_energy - gt_energy),
            "energy_diff (pred-calc_energy)": abs(pred_energy - calc_energy),
            "energy_diff (gt-calc_energy)": abs(gt_energy - calc_energy),

            "pred_force": pred_forces,
            "gt_force": gt_forces,
            "calc_force": calc_forces,

            "forces_diff l2 (pred-gt)": abs(pred_forces - gt_forces).mean(),
            "forces_diff l2 (pred-calc_forces)": abs(pred_forces - calc_forces).mean(),
            "forces_diff l2 (gt-calc_forces)": abs(gt_forces - calc_forces).mean(),

            "pred_force_norm": pred_forces_norm,
            "gt_force_norm": gt_forces_norm,
            "calc_force_norm": calc_forces_norm,

            "pred_force_norm_diff (pred-gt)": abs(pred_forces_norm - gt_forces_norm).mean(),
            "pred_force_norm_diff (pred-calc_forces)": abs(pred_forces_norm - calc_forces_norm).mean(),
            "gt_force_norm_diff (gt-calc_forces)": abs(gt_forces_norm - calc_forces_norm).mean(),

            "orbital_coeff_similarity (pred-gt)": torch.cosine_similarity(torch.tensor(pred_mo_coeff), torch.tensor(gt_mo_coeff), dim=1).abs().mean(),
            "orbital_coeff_similarity (pred-calc)": torch.cosine_similarity(torch.tensor(pred_mo_coeff), torch.tensor(calc_mo_coeff), dim=1).abs().mean(),
            "orbital_coeff_similarity (gt-calc)": torch.cosine_similarity(torch.tensor(gt_mo_coeff), torch.tensor(calc_mo_coeff), dim=1).abs().mean(),

            "occupied_orbital_energy_mae (pred-gt)": np.abs(pred_mo_energy_occ - gt_mo_energy_occ).mean(),
            "occupied_orbital_energy_mae (pred-calc)": np.abs(pred_mo_energy_occ - calc_mo_energy_occ).mean(),
            "occupied_orbital_energy_mae (gt-calc)": np.abs(gt_mo_energy_occ - calc_mo_energy_occ).mean(),
        }
    except Exception as e:
        print(f"Error processing molecule {data_index}: {str(e)}")
        result = {
            "error_count": data_index,
        }
    return result



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str, default="/data/qhflow-mlff/all_checkpoints/md17-salicylic_acid/output_dump")
    parser.add_argument("--pred_prefix", type=str, default="pred_")
    parser.add_argument("--gt_prefix", type=str, default="gt_")
    parser.add_argument("--num_procs", type=int, default=1)
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()

    dir_path = args.dir_path
    pred_prefix = args.pred_prefix
    list_pred_paths = glob.glob(os.path.join(dir_path, f"{pred_prefix}*.pt"))
    # sort by file name
    list_pred_paths.sort()

    gt_prefix = args.gt_prefix
    list_gt_paths = glob.glob(os.path.join(dir_path, f"{gt_prefix}*.pt"))
    # sort by file name
    list_gt_paths.sort()

    # apply multiprocessing to process list_pred_paths and list_gt_paths
    num_procs = args.num_procs

    # Create list of (pred_path, gt_path) tuples
    file_pairs = list(zip(list_pred_paths, list_gt_paths))
    if args.debug:
        file_pairs = file_pairs[:20]

    print(f"Processing {len(file_pairs)} molecules with {num_procs} processes...")

    if  num_procs == 1:
        iter_bar = tqdm(file_pairs, desc="Processing molecules")
        results = [process_single_molecule(pred_path, gt_path) for pred_path, gt_path in iter_bar]
    else:
        # Process with multiprocessing
        with Pool(processes=num_procs) as pool:
            results = list(tqdm(
                pool.starmap(process_single_molecule, file_pairs),
            total=len(file_pairs),
            desc="Processing molecules"
        ))

    print(f"\nCompleted processing {len(results)} molecules")

    # Convert keys to list and remove non-numeric keys
    keys = list(results[0].keys())
    keys_to_remove = ["data_index", "pred_force", "gt_force", "calc_force", "pred_force_norm", "gt_force_norm", "calc_force_norm"]
    keys = [k for k in keys if k not in keys_to_remove]
    keys.append("error_count")

    evaluation_result = {key: [] for key in keys}
    for result in results:
        for key in result.keys():
            if key not in keys_to_remove:
                evaluation_result[key].append(result[key])

    for key in keys:
        if key == "error_count":
            evaluation_result[key] = len(evaluation_result[key])
        else:
            evaluation_result[key] = float(np.mean(evaluation_result[key]))


    # Print evaluation results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    for key, value in evaluation_result.items():
        print(f"{key}: {value}")
    print("="*80)

    dir_dir_path = os.path.dirname(dir_path)
    # Save evaluation results
    output_file = os.path.join(dir_dir_path, "evaluation_results.json")
    with open(output_file, "w") as f:
        json.dump(evaluation_result, f, indent=4)
    print(f"\nEvaluation results saved to: {output_file}")