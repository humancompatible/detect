import logging
import pandas as pd
import numpy as np
from typing import Any, Callable, Dict, List, Tuple

from humancompatible.detect.binarizer.Binarizer import Bin
from humancompatible.detect.methods.one_rule import OneRule
from humancompatible.detect.prepare import prepare_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def most_biased_subgroup(
    X: pd.DataFrame,
    y: pd.DataFrame,
    protected_list: List[str] | None = None,
    continuous_list: List[str] | None = None,
    fp_map: Dict[str, Callable[Any, int]] | None = None,
    seed: int | None = None,
    n_samples: int = 1_000_000,
    method: str = "MSD",
    method_kwargs: Dict[str, Any] | None = None,
) -> List[Tuple[int, Bin]]:
    
    if seed is not None:
        logger.info(f"Seeding the run with seed={seed}")
        np.random.seed(seed)
    
    if continuous_list is None:
        continuous_list = []
    if fp_map is None:
        fp_map = {}
    if method_kwargs is None:
        method_kwargs = {}

    binarizer, X_prot, y = prepare_dataset(
        X,
        y,
        n_samples,
        protected_attrs=protected_list,
        continuous_feats=continuous_list,
        feature_processing=fp_map,
    )

    X_bin = binarizer.encode(X_prot, include_binary_negations=True)
    y_bin = binarizer.encode_y(y)

    if method == "MSD":
        indices = get_conjuncts_MSD(
            X_bin,
            y_bin,
            **method_kwargs
        )


        mio = OneRule()
        indices, _ = mio.find_rule(
            X_bin, y_bin, **method_kwargs
        )
    else:
        raise ValueError(f"Method '{method}' is not supported.")

    encodings = binarizer.get_bin_encodings(include_binary_negations=True)
    feats = binarizer.data_handler.features
    rule = [(feats.index(encodings[i].feature), encodings[i]) for i in indices]

    return rule
