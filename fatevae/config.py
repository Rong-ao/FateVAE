"""Central paths and constants for the FateVAE project.

All dataset locations live outside the repo (read-only inputs); all outputs
are written under RESULTS_DIR inside the repo.
"""
from pathlib import Path

import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "models"

for _d in (RESULTS_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- raw inputs
DATA_ROOT = Path(
    os.environ.get(
        "FATEVAE_DATA_ROOT",
        "data",
    )
)

# Processed larval disc objects (AnnData, cells x genes). NOTE: the
# `layers['counts']` inside these files is a corrected/normalized matrix,
# NOT raw UMIs - raw counts come from the *_filtered_raw.h5ad siblings.
DMEL_H5AD = DATA_ROOT / "result" / "Dmel" / "Dmel_male_scRNA.h5ad"
DMEL_RAW_H5AD = DATA_ROOT / "result" / "Dmel" / "Dmel_male_scRNA_filtered_raw.h5ad"
DSUZ_H5AD = DATA_ROOT / "result" / "Dsuz" / "Dsuz_male_scRNA.h5ad"
DSUZ_RAW_H5AD = DATA_ROOT / "result" / "Dsuz" / "Dsuz_male_scRNA_filtered_raw.h5ad"

# Larva + adult merged object with terminal-fate annotations (prospective
# validation reference)
DMEL_MERGED_H5AD = DATA_ROOT / "result" / "Dmel" / "merged_scRNA.h5ad"

# pySCENIC outputs (loom: regulon AUC per cell + regulon gene memberships)
DMEL_SCENIC_LOOM = DATA_ROOT / "result" / "Dmel" / "final.loom"
DSUZ_SCENIC_LOOM = DATA_ROOT / "result" / "Dsuz" / "final.loom"

# cisTarget motif ranking databases (for future GRN re-inference)
DMEL_MOTIF_RANKINGS = DATA_ROOT / "Dmel_base" / "dm6_v10_clust.genes_vs_motifs.rankings.feather"

# Adult terminal-fate categories in merged_scRNA.h5ad used for the
# prospective validation mapping (see io.loaders.map_larva_to_adult_fates)
ADULT_FATE_GROUPS = {
    "accessory_gland": [
        "male accessory gland main cell",
        "male accessory gland secondary cell",
    ],
    "ejaculatory_bulb": [
        "ejaculatory bulb",
        "ejaculatory bulb epithelium",
    ],
    "ejaculatory_duct": [
        "anterior ejaculatory duct",
        "secretory cells of the male reproductive tract",
    ],
    "seminal_vesicle": ["seminal vesicle"],
    # 'unknown', 'spermatid', hemocytes etc. are contaminants / non-disc
    # populations and are excluded from fate anchoring
}

# Random seed shared across the project
SEED = 0


def n_cpus() -> int:
    return max(1, len(os.sched_getaffinity(0)))
