import os
import glob
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import hydra
from src.data.components.calcium_dataset import CalciumDataset
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class CalciumDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir, 
        split_mode: str = "random", # random, folder, file, loo
        
        
        loo_index: int = 0, 
        loo_target: Optional[str] = None,
        
        target_datasets: Optional[List[str]] = None,
        train_files: Optional[List[str]] = None,
        test_files: Optional[List[str]] = None,
        omission_dict: Dict[str, List[int]] = None,
        train_test_split: Tuple[float, float] = (0.8, 0.2),

        # --- Trans Params ---
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        seed_value: int = 42,
        sampling_rate: float = 30.0,
        smoothing: float = 1.0,
        segment_len: int = 100,
        context_left: int = 30,
        context_right: int = 30,
        overlap_size: int = 2,
        feature_mode: str = "dual",
        resample: bool = False,
        causal_kernel: bool = False,
        denoise_method: str = "none",
        denoise_params: Optional[Dict] = None, 
        ephys_fs: float = 1e4,
        global_norm: str = "none",   # "none" | "zscore" | "minmax"
        bad_data_json: Optional[str] = None,  # path to bad_data.json for trial-level filtering
        target_clip: float = None,  # if set, clip spike targets to [0, target_clip] (e.g. 1.0 for OGB1)
        use_simulated_data: bool = False,
        simulated_data_dir: str = "Simulated_Calcium_Data",
        extra_simulated_data_dirs: list = [],  # additional simulated data dirs to include when use_simulated_data=True
        # --- Few-shot adaptation ---
        fewshot_k: int = 0,            # 0 = disabled; >0 = support set size (neurons)
        fewshot_pool_k: int = 0,       # if >0: reserve this many test neurons as support pool; query = test \ pool
        fewshot_seed: int = 0,         # random seed for neuron sampling
        fewshot_val_ratio: float = 0.15,  # fraction of remaining test neurons for val
        # --- Data scaling (rebuttal) ---
        train_subsample_fraction: float = 1.0,  # subsample resolved train folders (real data)
        train_subsample_seed: int = 42,
        train_max_indicator_families: int = 0,  # 0 = no cap; else keep first K dye families
        max_sim_dirs: int = -1,  # -1 = all sim dirs when use_simulated_data; else cap count
        
        **kwargs 
    ) -> None:
        
        super().__init__()
        self.save_hyperparameters(logger=False)  
        self.data_train: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        self.omission_dict = omission_dict if omission_dict is not None else {}
        self.seed_value = seed_value

        # bad_data trial-level filtering
        self.bad_data_dict = {}
        if self.hparams.bad_data_json:
            json_path = Path(self.hparams.bad_data_json)
            if not json_path.is_absolute():
                json_path = Path(os.getcwd()) / json_path
            if json_path.exists():
                with open(json_path, 'r') as f:
                    raw = json.load(f)
                
                for ds_name, files in raw.items():
                    self.bad_data_dict[ds_name] = {}
                    for fname, trials in files.items():
                        if fname == 'comment':
                            continue
                        
                        self.bad_data_dict[ds_name][fname] = set(t - 1 for t in trials)
                log.info(f"Loaded bad_data filter from {json_path}: "
                         f"{sum(len(v) for v in self.bad_data_dict.values())} file entries")
            else:
                log.warning(f"bad_data_json not found: {json_path}")
        
        
        valid_modes = ["random", "folder", "file", "loo"]
        if self.hparams.split_mode not in valid_modes:
            raise ValueError(f"split_mode must be one of {valid_modes}, got {self.hparams.split_mode}")

        data_root = Path(self.hparams.data_dir)
        if not data_root.exists():
            raise FileNotFoundError(f"Data directory not found at: {data_root}")
        self.data_root = data_root


    def setup(self, stage: Optional[str] = None) -> None:
        ""



        if self.data_train and self.data_test:
            return

        mode = self.hparams.split_mode

        
        if mode == "random":
            self._setup_random_mode(self.data_root)
        elif mode == "folder":
            self._setup_folder_mode(self.data_root)
        elif mode == "file":
            self._setup_file_mode(self.data_root)
        elif mode == "loo":
            self._setup_global_loo_mode(self.data_root)

        log.info(f"Setup Complete. Train sets: {len(self.data_train) if self.data_train else 0}, "
                 f"Test sets: {len(self.data_test) if self.data_test else 0}")

        # --- Few-shot neuron-level re-split ---
        if self.hparams.fewshot_k > 0:
            self._apply_fewshot_split()



    def _setup_random_mode(self, data_root: Path):
        ""



        working_folders = self._get_working_folders(data_root)
        total_count = len(working_folders)
        train_count = int(total_count * self.hparams.train_test_split[0])
        
        generator = torch.Generator().manual_seed(self.seed_value)
        indices = torch.randperm(total_count, generator=generator).tolist()
        
        train_paths = [working_folders[i] for i in indices[:train_count]]
        test_paths = [working_folders[i] for i in indices[train_count:]]

        self.data_train = self._create_dataset(train_paths)
        self.data_test = self._create_dataset(test_paths)


    def _setup_folder_mode(self, data_root: Path):
        ""







        train_pats = self.hparams.train_files
        test_pats = self.hparams.test_files
        
        working_folders = self._get_working_folders(data_root)

        train_folders = self._filter_folders(working_folders, train_pats, include=True) if train_pats else []
        test_folders = self._filter_folders(working_folders, test_pats, include=True) if test_pats else []

        
        if (
            len(train_folders) == 1
            and not test_pats
            and not getattr(self.hparams, "train_only", False)
        ):
            log.info(">> Folder Mode: Single train folder detected with no test patterns.")
            self._setup_single_folder_loo(train_folders[0])
            return

        
        if train_pats and not test_pats:
            if getattr(self.hparams, "train_only", False):
                test_folders = []
            else:
                test_folders = self._filter_folders(working_folders, train_pats, include=False)
        elif test_pats and not train_pats:
            train_folders = self._filter_folders(working_folders, test_pats, include=False)

        train_folders = self._apply_train_scaling(train_folders)

        # Append simulated data folders to training set if enabled
        if self.hparams.use_simulated_data:
            all_sim = [self.hparams.simulated_data_dir] + list(self.hparams.extra_simulated_data_dirs)
            cap = int(getattr(self.hparams, "max_sim_dirs", -1) or -1)
            sim_names = all_sim if cap < 0 else all_sim[:cap]
            for sim_name in sim_names:
                sim_dir = self.data_root / sim_name
                if sim_dir.exists() and sim_dir.is_dir():
                    train_folders.append(str(sim_dir))
                    log.info(f">> Added simulated data folder: {sim_dir}")
                else:
                    log.warning(f"Simulated data dir not found: {sim_dir}")

        self.data_train = self._create_dataset(train_folders)
        self.data_test = self._create_dataset(test_folders)


    def _setup_file_mode(self, data_root: Path):
        ""



        train_files = self.hparams.train_files
        test_files = self.hparams.test_files or []
        
        if not train_files:
             raise ValueError("In 'file' mode, 'train_files' must be provided!")

        train_paths = self._resolve_file_paths(data_root, train_files)
        test_paths = self._resolve_file_paths(data_root, test_files)

        self.data_train = self._create_dataset(train_paths)
        self.data_test = self._create_dataset(test_paths)


    def _setup_global_loo_mode(self, data_root: Path):
        ""




        working_folders = self._get_working_folders(data_root)
        total_files = len(working_folders)
        
        if total_files == 0:
            raise RuntimeError("No valid data folders found for LOO mode!")

        test_target_path = None
        train_target_paths = []

        
        if hasattr(self.hparams, "loo_target") and self.hparams.loo_target is not None:
            target_name = self.hparams.loo_target
            
            for folder in working_folders:
                if os.path.basename(folder) == target_name:
                    test_target_path = folder
                else:
                    train_target_paths.append(folder)
                    
            if test_target_path is None:
                raise ValueError(f"loo_target '{target_name}' not found in {data_root}!")
                
            log.info(f">> LOO Mode: Explicitly selected '{target_name}' as test set by name.")


        
        else:
            working_folders = sorted(working_folders)
            idx = self.hparams.loo_index
            if idx < 0 or idx >= total_files:
                raise ValueError(f"loo_index {idx} out of bounds for {total_files} folders!")
                
            test_target_path = working_folders[idx]
            train_target_paths = working_folders[:idx] + working_folders[idx+1:]
            
            log.info(f">> LOO Mode: Selected '{os.path.basename(test_target_path)}' as test set by index {idx}.")


        self.data_train = self._create_dataset(train_target_paths)
        self.data_test = self._create_dataset([test_target_path])


    def _setup_single_folder_loo(self, target_folder: str):
        ""
        log.info(f">> Applying internal Leave-One-Mat-Out for folder: {os.path.basename(target_folder)}")
        folder_name = os.path.basename(target_folder)
        
        all_files = []
        all_files.extend(glob.glob(os.path.join(target_folder, '*.nwb')))
        all_files.extend(glob.glob(os.path.join(target_folder, '*_mini.mat')))

        bad_indices = self.hparams.omission_dict.get(folder_name, [])
        available_files = [f for i, f in enumerate(all_files) if i not in bad_indices]
        
        if not available_files:
            raise RuntimeError(f"Folder {folder_name} has no valid files!")
        
        random.seed(self.seed_value)
        test_mat_path = random.choice(available_files)
        train_mat_paths = [f for f in available_files if f != test_mat_path]
        
        self.data_train = self._create_dataset(train_mat_paths)
        self.data_test = self._create_dataset([test_mat_path])


    def _apply_fewshot_split(self):
        """Re-split test set into support (train) + query (test) at neuron (.mat) level.

        Called after normal setup (folder/loo mode) when fewshot_k > 0.
        Uses hparams.test_files (folder names from LOO config) to enumerate .mat files,
        then splits them into support (new train) and query (new test).

        Stratified sampling: each test folder (sub-domain) gets at least 1 neuron
        in the support set, remaining quota allocated proportionally.
        """
        k = self.hparams.fewshot_k
        seed = self.hparams.fewshot_seed

        # Collect .mat files grouped by folder (sub-domain)
        test_file_names = self.hparams.test_files or []
        if not test_file_names:
            raise RuntimeError("Cannot apply few-shot split: hparams.test_files is empty")

        folder_mats = {}  # folder_name -> list of mat paths
        for name in test_file_names:
            folder = self.data_root / name
            if folder.is_dir():
                mats = sorted(glob.glob(os.path.join(str(folder), '*_mini.mat')))
                if not mats and "Simulated" in name:
                    mats = sorted(glob.glob(os.path.join(str(folder), '*.mat')))
                if mats:
                    folder_mats[name] = mats
            elif folder.with_suffix('.mat').is_file():
                folder_mats[name] = [str(folder.with_suffix('.mat'))]

        n_folders = len(folder_mats)
        n_total = sum(len(v) for v in folder_mats.values())
        if n_total == 0:
            raise RuntimeError("Few-shot split: no .mat files found in test paths")
        if k >= n_total:
            raise ValueError(f"fewshot_k={k} >= total test neurons={n_total}")
        if k < n_folders:
            log.warning(f"fewshot_k={k} < n_folders={n_folders}, "
                        f"cannot guarantee coverage of all sub-domains")

        rng = random.Random(seed)

        pool_k = int(getattr(self.hparams, "fewshot_pool_k", 0) or 0)
        if pool_k > 0:
            pool_size = min(pool_k, n_total - 1)
            if k > pool_size:
                raise ValueError(
                    f"fewshot_k={k} exceeds fewshot_pool_k={pool_size} "
                    f"(need at least one query neuron)"
                )
            pool_files = self._fewshot_stratified_pick(folder_mats, pool_size, rng)
            pool_set = set(pool_files)
            support_files = pool_files[:k]
            remaining_files = [
                m for mats in folder_mats.values() for m in mats if m not in pool_set
            ]
            log.info(
                f">> Few-shot (fixed pool): pool={pool_size}, k={k}, seed={seed}, "
                f"support={len(support_files)}, query={len(remaining_files)}"
            )
            self.data_train = self._create_dataset(support_files)
            self.data_test = self._create_dataset(remaining_files)
            return

        # --- Legacy: support + query partition without fixed pool ---
        support_files = []
        remaining_files = []

        # Shuffle within each folder first
        folder_pools = {}
        for name in sorted(folder_mats.keys()):
            pool = list(folder_mats[name])
            rng.shuffle(pool)
            folder_pools[name] = pool

        if k >= n_folders:
            # Phase 1: 1 per folder guaranteed
            leftover_quota = k - n_folders
            for name, pool in folder_pools.items():
                support_files.append(pool[0])
                folder_pools[name] = pool[1:]

            # Phase 2: distribute remaining quota proportionally
            if leftover_quota > 0:
                sizes = {name: len(pool) for name, pool in folder_pools.items()}
                total_remaining = sum(sizes.values())
                # Proportional allocation with largest-remainder method
                raw = {name: (sz / total_remaining * leftover_quota) if total_remaining > 0 else 0
                       for name, sz in sizes.items()}
                alloc = {name: int(v) for name, v in raw.items()}
                remainders = {name: raw[name] - alloc[name] for name in raw}
                shortfall = leftover_quota - sum(alloc.values())
                # Give extra slots to folders with largest remainders
                for name in sorted(remainders, key=remainders.get, reverse=True):
                    if shortfall <= 0:
                        break
                    if alloc[name] < sizes[name]:  # don't exceed available
                        alloc[name] += 1
                        shortfall -= 1

                for name, pool in folder_pools.items():
                    n_take = min(alloc.get(name, 0), len(pool))
                    support_files.extend(pool[:n_take])
                    remaining_files.extend(pool[n_take:])
            else:
                for pool in folder_pools.values():
                    remaining_files.extend(pool)
        else:
            # k < n_folders: pick k folders randomly, 1 neuron each
            chosen = rng.sample(sorted(folder_pools.keys()), k)
            for name, pool in folder_pools.items():
                if name in chosen:
                    support_files.append(pool[0])
                    remaining_files.extend(pool[1:])
                else:
                    remaining_files.extend(pool)

        # Log per-folder allocation
        support_set = set(support_files)
        per_folder = {}
        for name, mats in folder_mats.items():
            n_sup = sum(1 for m in mats if m in support_set)
            per_folder[name] = f"{n_sup}/{len(mats)}"
        log.info(f">> Few-shot split (stratified): k={k}, seed={seed}, "
                 f"total={n_total}, support={len(support_files)}, query={len(remaining_files)}")
        log.info(f">> Per-folder allocation: {per_folder}")

        # Recreate datasets from individual .mat file paths
        self.data_train = self._create_dataset(support_files)
        self.data_test = self._create_dataset(remaining_files)

    def _fewshot_stratified_pick(
        self,
        folder_mats: Dict[str, List[str]],
        k: int,
        rng: random.Random,
    ) -> List[str]:
        """Stratified pick of k neuron .mat paths across test folders."""
        n_folders = len(folder_mats)
        n_total = sum(len(v) for v in folder_mats.values())
        if k >= n_total:
            raise ValueError(f"fewshot pick k={k} >= total test neurons={n_total}")
        if k < n_folders:
            log.warning(
                f"fewshot pick k={k} < n_folders={n_folders}, "
                "cannot guarantee coverage of all sub-domains"
            )

        folder_pools: Dict[str, List[str]] = {}
        for name in sorted(folder_mats.keys()):
            pool = list(folder_mats[name])
            rng.shuffle(pool)
            folder_pools[name] = pool

        support_files: List[str] = []
        if k >= n_folders:
            leftover_quota = k - n_folders
            for name, pool in folder_pools.items():
                support_files.append(pool[0])
                folder_pools[name] = pool[1:]

            if leftover_quota > 0:
                sizes = {name: len(pool) for name, pool in folder_pools.items()}
                total_remaining = sum(sizes.values())
                raw = {
                    name: (sz / total_remaining * leftover_quota) if total_remaining > 0 else 0
                    for name, sz in sizes.items()
                }
                alloc = {name: int(v) for name, v in raw.items()}
                remainders = {name: raw[name] - alloc[name] for name in raw}
                shortfall = leftover_quota - sum(alloc.values())
                for name in sorted(remainders, key=remainders.get, reverse=True):
                    if shortfall <= 0:
                        break
                    if alloc[name] < sizes[name]:
                        alloc[name] += 1
                        shortfall -= 1
                for name, pool in folder_pools.items():
                    n_take = min(alloc.get(name, 0), len(pool))
                    support_files.extend(pool[:n_take])
        else:
            chosen = rng.sample(sorted(folder_pools.keys()), k)
            for name, pool in folder_pools.items():
                if name in chosen:
                    support_files.append(pool[0])

        return support_files


    '''
    Legacy implementation retained for reference.
    def _get_working_folders(self, data_root: Path) -> List[str]:
        working_folders = []
        
        if self.hparams.target_datasets:
            for ds_rel_path in self.hparams.target_datasets:
                folder_path = data_root / ds_rel_path
                
                if not folder_path.exists() or not folder_path.is_dir():
                    log.warning(f"Target path not found or is not a directory: {folder_path}")
                    continue
                
                sub_dirs = [x for x in folder_path.iterdir() if x.is_dir() and not x.name.startswith(".")]
                
                if sub_dirs:
                    working_folders.extend([str(x) for x in sub_dirs])
                else:
                    working_folders.append(str(folder_path))

            working_folders = sorted(list(set(working_folders)))
        
        else:
            for entry in sorted(data_root.iterdir()):
                if entry.is_dir() and not entry.name.startswith("."):
                    working_folders.append(str(entry))
                    
        if not working_folders:
            raise RuntimeError(f"No valid dataset folders found in {data_root}!")
            
        return working_folders
    '''

    def _get_working_folders(self, data_root: Path) -> List[str]:
        ""




        
        working_folders = set()       
        
        search_roots = []
        if self.hparams.target_datasets:
            for ds_rel_path in self.hparams.target_datasets:
                folder_path = data_root / ds_rel_path
                if folder_path.exists() and folder_path.is_dir():
                    search_roots.append(folder_path)
                else:
                    log.warning(f"Target path not found or not a dir: {folder_path}")
        else:
            search_roots = [data_root]

        for root in search_roots:
            for file_path in root.rglob("*.nwb"):
                working_folders.add(str(file_path.parent))
            for file_path in root.rglob("*.mat"):
                working_folders.add(str(file_path.parent))
                
        if not working_folders:
            raise RuntimeError(f"No valid data folders (containing .nwb or .mat files) found in specified paths!")
            
        return sorted(list(working_folders))


    def _create_dataset(self, paths: List[str]) -> CalciumDataset:
        if not paths: 
            log.warning("Empty paths list, returning None dataset")
            return None
        return CalciumDataset(
            data_paths=paths,
            omission_dict=self.hparams.omission_dict,
            sampling_rate=self.hparams.sampling_rate,
            smoothing=self.hparams.smoothing,
            segment_len=self.hparams.segment_len,
            context_left=self.hparams.context_left,
            context_right=self.hparams.context_right,
            overlap_size=self.hparams.overlap_size,
            resample=self.hparams.resample,
            causal_kernel=self.hparams.causal_kernel,
            denoise_method=self.hparams.denoise_method,
            denoise_params=self.hparams.denoise_params,
            feature_mode=self.hparams.feature_mode,
            ephys_fs=self.hparams.ephys_fs,
            global_norm=self.hparams.global_norm,
            bad_data_dict=self.bad_data_dict,
            target_clip=self.hparams.target_clip if self.hparams.target_clip else None,
        )


    def _resolve_file_paths(self, data_root: Path, file_inputs: List[str]) -> List[str]:
        resolved_paths = []
        for inp in file_inputs:
            direct_path = data_root / inp
            if direct_path.exists() and direct_path.is_file():
                resolved_paths.append(str(direct_path))
            else:
                abs_path = Path(inp)
                if abs_path.exists() and abs_path.is_file():
                    resolved_paths.append(str(abs_path))
                else:
                    log.warning(f"File not found: {inp}")
        return resolved_paths

    @staticmethod
    def _indicator_family(folder_name: str) -> str:
        """Dye / indicator token from folder name, e.g. DS14-GCaMP6s-m-V1 -> GCaMP6s."""
        parts = folder_name.split("-")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
        return folder_name

    def _apply_train_scaling(self, train_folders: List[str]) -> List[str]:
        """Subsample real train folders and/or cap indicator-family diversity."""
        folders = list(train_folders)
        max_fam = int(getattr(self.hparams, "train_max_indicator_families", 0) or 0)
        if max_fam > 0 and folders:
            by_fam: Dict[str, List[str]] = {}
            for f in folders:
                fam = self._indicator_family(os.path.basename(f))
                by_fam.setdefault(fam, []).append(f)
            kept = sorted(by_fam.keys())[:max_fam]
            folders = [f for fam in kept for f in sorted(by_fam[fam])]
            log.info(
                f">> Indicator scaling: {len(kept)} families, {len(folders)} folders"
            )
        frac = float(getattr(self.hparams, "train_subsample_fraction", 1.0) or 1.0)
        if 0.0 < frac < 1.0 and folders:
            seed = int(getattr(self.hparams, "train_subsample_seed", self.seed_value))
            by_fam: Dict[str, List[str]] = {}
            for f in folders:
                fam = self._indicator_family(os.path.basename(f))
                by_fam.setdefault(fam, []).append(f)
            selected: List[str] = []
            for fam in sorted(by_fam.keys()):
                fam_folders = sorted(by_fam[fam])
                n = max(1, int(round(len(fam_folders) * frac)))
                n = min(n, len(fam_folders))
                fam_seed = seed + sum(ord(c) for c in fam)
                rng = random.Random(fam_seed)
                selected.extend(rng.sample(fam_folders, n))
            folders = sorted(selected)
            log.info(
                f">> Real-data scaling (per-indicator): {len(folders)} folders "
                f"from {len(by_fam)} families (fraction={frac})"
            )
        return folders


    def _filter_folders(self, working_folder: List[str], folder_list: List[str], include: bool = True) -> List[str]:
        result = []
        matched_folders = set()

        for p in folder_list:
            folder_path = self.data_root / p
            if folder_path.exists() and folder_path.is_dir():
                matched_folders.add(str(folder_path))
            else:
                for folder in working_folder:
                    folder_name = os.path.basename(folder)
                    if p == folder_name or folder.endswith(p):
                        matched_folders.add(folder)

        if include:
            result = sorted(list(matched_folders))
        else:
            for folder in working_folder:
                is_match = folder in matched_folders
                if not include and not is_match:
                    result.append(folder)
            result = sorted(result)
        
        return result
    
        '''
        result = []
        for folder in pool:
            folder_name = os.path.basename(folder)
            is_match = any(p in folder_name for p in patterns)
            if (include and is_match) or (not include and not is_match):
                result.append(folder)
        return sorted(result)
        '''


    def train_dataloader(self) -> DataLoader:
        if not self.data_train: 
            log.warning("Train set empty!")
            raise RuntimeError("Train set empty!")
        persistent_workers = bool(self.hparams.persistent_workers) and self.hparams.num_workers > 0
        return DataLoader(self.data_train, 
                          batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers, 
                          shuffle=True, 
                          pin_memory=self.hparams.pin_memory,
                          persistent_workers=persistent_workers)


    def val_dataloader(self) -> DataLoader:
        if not self.data_test:
            return DataLoader([], batch_size=self.hparams.batch_size)
        persistent_workers = bool(self.hparams.persistent_workers) and self.hparams.num_workers > 0
        return DataLoader(self.data_test,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          shuffle=False,
                          pin_memory=self.hparams.pin_memory,
                          persistent_workers=persistent_workers)

    def test_dataloader(self) -> DataLoader:
        if not self.data_test: 
            return DataLoader([], batch_size=self.hparams.batch_size)
        persistent_workers = bool(self.hparams.persistent_workers) and self.hparams.num_workers > 0
        return DataLoader(self.data_test, 
                          batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers, 
                          shuffle=False, 
                          pin_memory=self.hparams.pin_memory,
                          persistent_workers=persistent_workers)
