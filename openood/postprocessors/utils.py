import importlib


# mapping: name -> (module_path, class_name)
_POSTPROCESSOR_REGISTRY = {
    # common / core methods (usually no exotic deps)
    "msp": ("openood.postprocessors.msp_postprocessor", "MSPPostprocessor"),
    "maxlogit": ("openood.postprocessors.maxlogit_postprocessor", "MaxLogitPostprocessor"),
    "odin": ("openood.postprocessors.odin_postprocessor", "ODINPostprocessor"),
    "mds": ("openood.postprocessors.mds_postprocessor", "MDSPostprocessor"),
    "ebo": ("openood.postprocessors.ebo_postprocessor", "EBOPostprocessor"),
    "energy": ("openood.postprocessors.energy_postprocessor", "EnergyPostprocessor"),
    "vim": ("openood.postprocessors.vim_postprocessor", "ViMPostprocessor"),

    # your Step5 method
    "neco": ("openood.postprocessors.neco_postprocessor", "NECOPostprocessor"),

    # optional methods below (may require extra deps; keep them here if you want)
    # "openmax": ("openood.postprocessors.openmax_postprocessor", "OpenMax"),
    # "adascale": ("openood.postprocessors.adascale_postprocessor", "AdaScalePostprocessor"),
    # ... you can add others later
}


def _load_postprocessor_class(name: str):
    """Load postprocessor class by name with lazy import."""
    if name not in _POSTPROCESSOR_REGISTRY:
        raise KeyError(
            f"Unknown postprocessor '{name}'. "
            f"Available: {sorted(_POSTPROCESSOR_REGISTRY.keys())}"
        )

    module_path, class_name = _POSTPROCESSOR_REGISTRY[name]
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        # Typically missing optional dependency
        raise ModuleNotFoundError(
            f"Failed to import postprocessor '{name}' from '{module_path}'. "
            f"This usually means an optional dependency is missing.\n"
            f"Original error: {repr(e)}"
        ) from e

    try:
        cls = getattr(mod, class_name)
    except AttributeError as e:
        raise AttributeError(
            f"Module '{module_path}' does not have class '{class_name}'."
        ) from e

    return cls


def get_postprocessor(config):
    """
    Factory used by OpenOOD evaluator.
    It reads config.postprocessor.name and returns an instance.
    """
    name = config.postprocessor.name
    cls = _load_postprocessor_class(name)
    return cls(config)
