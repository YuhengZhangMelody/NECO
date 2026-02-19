from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class NECOPostprocessor(BasePostprocessor):
    """
    NECO: Neural Collapse Inspired OOD Detection

    Score(x) = || P_k(z) || / (||z|| + eps)

    where z is the (optionally standardized) penultimate-layer feature vector,
    and P_k is the projection onto the top-k PCA components fitted on ID train
    features.

    Optionally, score can be multiplied by MaxLogit.
    """




    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # ====== ADD THIS: evaluator expects APS_mode ======
        # Some OpenOOD versions put APS_mode under config.postprocessor.APS_mode
        # If missing, default False.
        try:
            self.APS_mode = bool(self.config.postprocessor.APS_mode)
        except Exception:
            self.APS_mode = False
        # ================================================

        self.args = self.config.postprocessor.postprocessor_args

        self.neco_dim = int(self.args.neco_dim)
        self.use_scaler = bool(getattr(self.args, "use_scaler", True))
        self.multiply_maxlogit = bool(getattr(self.args, "multiply_maxlogit", False))
        self.eps = float(getattr(self.args, "eps", 1e-12))

        # ====== ADD THIS: hyperparam sweep list for APS ======
        # evaluator may read self.config.postprocessor.postprocessor_sweep.<...>
        # but different versions differ; we keep a local fallback list.
        self.neco_dim_list = None
        try:
            sweep = self.config.postprocessor.postprocessor_sweep
            # common key name in configs: neco_dim_list
            if hasattr(sweep, "neco_dim_list"):
                self.neco_dim_list = list(sweep.neco_dim_list)
        except Exception:
            pass
        # ====================================================

        self.setup_flag = False
        # ====== REQUIRED by some OpenOOD evaluators (APS state) ======
        self.hyperparam_search_done = False
        self.best_hyperparam = [self.neco_dim]
        # ============================================================





    def get_sweep_list(self):
        # used by some OpenOOD versions
        if self.neco_dim_list is not None:
            return self.neco_dim_list
        return [self.neco_dim]


    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if self.setup_flag:
            return

        net.eval()

        # For optional max-logit (and consistent head access across networks)
        self.w, self.b = net.get_fc()

        # 1) Extract ID train features
        feats = []
        with torch.no_grad():
            for batch in tqdm(
                id_loader_dict["train"],
                desc="Setup (NECO): extracting ID train features",
                position=0,
                leave=True,
            ):
                data = batch["data"].cuda().float()
                _, feature = net(data, return_feature=True)
                feats.append(feature.cpu().numpy())

        feature_id_train = np.concatenate(feats, axis=0).astype(np.float32)

        # 2) (Optional) StandardScaler
        if self.use_scaler:
            self.scaler = StandardScaler()
            feature_id_train_proc = self.scaler.fit_transform(feature_id_train)
        else:
            self.scaler = None
            feature_id_train_proc = feature_id_train

        # 3) Fit PCA (full components)
        self.pca = PCA(n_components=feature_id_train_proc.shape[1])
        self.pca.fit(feature_id_train_proc)

        self.setup_flag = True

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        net.eval()

        logits, feature = net(data, return_feature=True)
        pred = logits.argmax(1)

        feat = feature.detach().cpu().numpy().astype(np.float32)

        if self.scaler is not None:
            feat_proc = self.scaler.transform(feat)
        else:
            feat_proc = feat

        # PCA projection, take first k dims
        reduced_all = self.pca.transform(feat_proc)
        reduced_k = reduced_all[:, : self.neco_dim]

        # NECO score: ||proj_k|| / ||full||
        ratio = norm(reduced_k, axis=1) / (norm(feat_proc, axis=1) + self.eps)
        score = ratio

        # Optional: multiply by MaxLogit (some architectures)
        if self.multiply_maxlogit:
            maxlogit = logits.detach().cpu().numpy().max(axis=1)
            score = score * maxlogit

        return pred, torch.from_numpy(score)

    # For OpenOOD APS (hyperparam search)
    def set_hyperparam(self, hyperparam: list):
        self.neco_dim = int(hyperparam[0])

    def get_hyperparam(self):
        return self.neco_dim

    def hyperparam_search(self, net, id_loader_dict, ood_loader_dict=None):
        """
        Automatic hyperparameter search (APS) for NECO.
        We choose neco_dim that maximizes AUROC on ID validation vs a proxy OOD set.

        If your pipeline provides an explicit OOD validation loader, use it.
        Otherwise, we fall back to using (ID val) vs (ID train shuffled) which is not ideal
        but allows APS to run without extra assumptions.

        After search, sets:
          - self.neco_dim
          - self.best_hyperparam
          - self.hyperparam_search_done = True
        """
        # If APS not enabled, just mark done and return
        if not getattr(self, "APS_mode", False):
            self.hyperparam_search_done = True
            self.best_hyperparam = [self.neco_dim]
            return

        # Ensure NECO is set up (PCA/scaler fitted on ID train)
        self.setup(net, id_loader_dict, ood_loader_dict)

        # Candidate list
        cand_list = self.neco_dim_list if self.neco_dim_list is not None else [self.neco_dim]

        # Loaders:
        # - ID validation loader is usually provided as id_loader_dict['val']
        id_val_loader = id_loader_dict.get("val", None) or id_loader_dict.get("test", None)
        if id_val_loader is None:
            # No val/test provided -> cannot do meaningful APS; pick current
            self.hyperparam_search_done = True
            self.best_hyperparam = [self.neco_dim]
            return

        # Try to find an OOD validation loader in ood_loader_dict
        ood_val_loader = None
        if ood_loader_dict is not None:
            # common keys used in OpenOOD: 'val', 'nearood', 'farood', etc.
            for k in ["val", "nearood", "farood", "ood_val", "csid", "ood"]:
                if k in ood_loader_dict:
                    ood_val_loader = ood_loader_dict[k]
                    break
            # Sometimes dict is nested: {'val': {'tin': loader, ...}}
            if isinstance(ood_val_loader, dict):
                # take first available
                ood_val_loader = next(iter(ood_val_loader.values()))

        # Fallback if no OOD val loader exists
        # (APS will still run, but it becomes a weak proxy)
        if ood_val_loader is None:
            ood_val_loader = id_loader_dict["train"]

        # Collect scores for a given neco_dim
        def collect_scores(loader, is_ood: bool):
            all_scores = []
            all_labels = []
            net.eval()
            with torch.no_grad():
                for batch in loader:
                    x = batch["data"].cuda().float()
                    _, s = self.postprocess(net, x)
                    all_scores.append(s.detach().cpu().numpy())
                    all_labels.append(np.ones(len(s)) if is_ood else np.zeros(len(s)))
            return np.concatenate(all_scores), np.concatenate(all_labels)

        # We'll need roc_auc_score
        try:
            from sklearn.metrics import roc_auc_score
        except Exception as e:
            raise ModuleNotFoundError(
                "scikit-learn is required for NECO APS search (roc_auc_score). "
                "Install it via: pip install scikit-learn"
            ) from e

        # Pre-collect ID and OOD data once per candidate? (scores depend on neco_dim, so no)
        best_auc = -1.0
        best_dim = self.neco_dim

        for dim in cand_list:
            self.neco_dim = int(dim)

            id_scores, id_labels = collect_scores(id_val_loader, is_ood=False)
            ood_scores, ood_labels = collect_scores(ood_val_loader, is_ood=True)

            scores = np.concatenate([id_scores, ood_scores])
            labels = np.concatenate([id_labels, ood_labels])

            auc = roc_auc_score(labels, scores)

            if auc > best_auc:
                best_auc = auc
                best_dim = int(dim)

        self.neco_dim = best_dim
        self.best_hyperparam = [best_dim]
        self.hyperparam_search_done = True
