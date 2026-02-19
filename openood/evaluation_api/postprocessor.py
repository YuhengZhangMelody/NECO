import importlib
import os
from omegaconf import OmegaConf

# name -> (module_path, class_name)
_POSTPROCESSOR_REGISTRY = {
    # ====== Common OOD scores you likely use in TP ======
    "msp": ("openood.postprocessors.msp_postprocessor", "MSPPostprocessor"),
    "maxlogit": ("openood.postprocessors.maxlogit_postprocessor", "MaxLogitPostprocessor"),
    "odin": ("openood.postprocessors.odin_postprocessor", "ODINPostprocessor"),
    "mds": ("openood.postprocessors.mds_postprocessor", "MDSPostprocessor"),
    "ebo": ("openood.postprocessors.ebo_postprocessor", "EBOPostprocessor"),
    "energy": ("openood.postprocessors.energy_postprocessor", "EnergyPostprocessor"),
    "vim": ("openood.postprocessors.vim_postprocessor", "ViMPostprocessor"),

    # ====== Step5 ======
    "neco": ("openood.postprocessors.neco_postprocessor", "NECOPostprocessor"),

    # If you need more, add them here later.
    # "ash": ("openood.postprocessors.ash_postprocessor", "ASHPostprocessor"),
    # "react": ("openood.postprocessors.react_postprocessor", "ReActPostprocessor"),
    # "openmax": ("openood.postprocessors.openmax_postprocessor", "OpenMax"),
}


def _load_class(name: str):
    if name not in _POSTPROCESSOR_REGISTRY:
        raise KeyError(
            f"Unknown postprocessor '{name}'. "
            f"Available: {sorted(_POSTPROCESSOR_REGISTRY.keys())}"
        )
    module_path, class_name = _POSTPROCESSOR_REGISTRY[name]
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Failed to import postprocessor '{name}' from '{module_path}'. "
            f"Likely missing an optional dependency.\n"
            f"Original error: {repr(e)}"
        ) from e

    try:
        return getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(
            f"Module '{module_path}' does not define '{class_name}'."
        ) from e


def get_postprocessor(config_root, postprocessor_name: str, id_data_name: str = None):
    """
    Compatible with OpenOOD versions where `config_root` is either:
    - an OmegaConf config object (has attribute `.postprocessor`), OR
    - a string path to the config folder (e.g., 'configs').

    We will lazy-import the target postprocessor class, and construct a minimal
    OmegaConf config if only a path is provided.
    """
    cls = _load_class(postprocessor_name)

    # Case 1: already a config object
    if not isinstance(config_root, str):
        return cls(config_root)

    # Case 2: config_root is a path string (e.g., 'configs')
    # Load postprocessor yaml:  <config_root>/postprocessors/<name>.yml
    pp_yml = os.path.join(config_root, "postprocessors", f"{postprocessor_name}.yml")
    if not os.path.exists(pp_yml):
        raise FileNotFoundError(
            f"Cannot find postprocessor config: {pp_yml}\n"
            f"Please make sure you created configs/postprocessors/{postprocessor_name}.yml"
        )

    pp_cfg = OmegaConf.load(pp_yml)

    # Build a minimal root config object that postprocessors expect
    # (Most postprocessors only access config.postprocessor.*)
    root_cfg = OmegaConf.create({"postprocessor": {}})

    # merge yaml content into root_cfg
    root_cfg = OmegaConf.merge(root_cfg, pp_cfg)

    # ensure the name is correct (some yml already has it, but force anyway)
    root_cfg.postprocessor.name = postprocessor_name

    return cls(root_cfg)
