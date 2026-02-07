"""
Benchmark Dataset Loader - Unified interface for all benchmark datasets.

Provides reproducible access to datasets from public sources:
- sklearn.datasets: Synthetic (Friedman) and classic (california_housing)
- OpenML: Grinsztajn benchmark suites (numerical/categorical regression)
- PMLB: Penn Machine Learning Benchmarks
- MoleculeNet: Chemistry datasets (ESOL, FreeSolv, Lipophilicity)

Usage:
    from benchmark_data import (
        load_dataset, list_datasets, get_dataset_info,
        get_benchmark_datasets
    )

    # List all datasets
    list_datasets()

    # Load a dataset with rich metadata
    X, y, meta = load_dataset('friedman1')
    meta.n_categorical           # Number of categorical features
    meta.categorical_cardinalities  # {col: cardinality}
    meta.feature_ranges          # {col: (min, max)}

Created: 14Jan26
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import warnings

# Cache directory for downloaded datasets
CACHE_DIR = Path(__file__).parent / '.cache'


# =============================================================================
# STRATA DEFINITIONS
# =============================================================================

STRATA = {
    'S1': {
        'name': 'Engineering/Simulation',
        'description': 'Smooth by design (control systems, simulations)',
        'expected_winner': 'ERBF, Chebyshev',
    },
    'S2': {
        'name': 'Behavioral/Social',
        'description': 'Threshold-prone (human decisions, behavior)',
        'expected_winner': 'Trees',
    },
    'S3': {
        'name': 'Physical/Chemical/Life',
        'description': 'Smooth with phase transitions (natural phenomena)',
        'expected_winner': 'ERBF, Chebyshev',
    },
    'S4': {
        'name': 'Economic/Pricing',
        'description': 'Threshold-heavy (explicit rules, market dynamics)',
        'expected_winner': 'Trees',
    },
}


# =============================================================================
# DATASET METADATA
# =============================================================================

@dataclass
class DatasetMeta:
    """Metadata about a dataset, computed after loading."""
    name: str
    source: str
    stratum: str  # S1-S4 classification

    # Size
    n_samples: int = 0
    n_features: int = 0

    # Feature types
    n_numerical: int = 0
    n_categorical: int = 0
    categorical_names: list = field(default_factory=list)
    categorical_cardinalities: dict = field(default_factory=dict)

    # Target properties
    target_min: float = 0.0
    target_max: float = 0.0
    target_mean: float = 0.0
    target_std: float = 0.0

    # Data quality
    n_missing: int = 0
    missing_fraction: float = 0.0

    # Feature scale info
    feature_ranges: dict = field(default_factory=dict)  # {name: (min, max)}
    feature_scales: dict = field(default_factory=dict)  # {name: std}

    # Additional info
    description: str = ""
    url: str = ""


def compute_metadata(X: pd.DataFrame, y: np.ndarray, name: str,
                     source: str, stratum: str, description: str = "",
                     url: str = "") -> DatasetMeta:
    """Compute rich metadata from loaded dataset."""
    meta = DatasetMeta(
        name=name,
        source=source,
        stratum=stratum,
        description=description,
        url=url,
        n_samples=len(X),
        n_features=X.shape[1],
    )

    # Identify categorical vs numerical
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    meta.n_categorical = len(cat_cols)
    meta.n_numerical = len(num_cols)
    meta.categorical_names = cat_cols

    # Categorical cardinalities
    for col in cat_cols:
        meta.categorical_cardinalities[col] = X[col].nunique()

    # Target statistics
    meta.target_min = float(np.nanmin(y))
    meta.target_max = float(np.nanmax(y))
    meta.target_mean = float(np.nanmean(y))
    meta.target_std = float(np.nanstd(y))

    # Missing values
    meta.n_missing = int(X.isna().sum().sum())
    meta.missing_fraction = meta.n_missing / (meta.n_samples * meta.n_features)

    # Feature ranges and scales (numerical only)
    for col in num_cols:
        col_data = X[col].dropna()
        if len(col_data) > 0:
            meta.feature_ranges[col] = (float(col_data.min()), float(col_data.max()))
            meta.feature_scales[col] = float(col_data.std())

    return meta


# =============================================================================
# DATASET REGISTRY
# =============================================================================

# Registry: {name: {loader_func, source, stratum, description, ...}}
DATASET_REGISTRY = {}


def register_dataset(name: str, loader_func, source: str, stratum: str,
                     description: str = "", url: str = "",
                     **kwargs):
    """
    Register a dataset in the registry.

    Parameters
    ----------
    name : str
        Dataset identifier
    loader_func : callable
        Function that returns (X, y)
    source : str
        Data source (sklearn, openml, pmlb, moleculenet, synthetic)
    stratum : str
        Benchmark stratum (S1-S4)
    description : str
        Brief description
    url : str
        Reference URL
    """
    DATASET_REGISTRY[name] = {
        'loader': loader_func,
        'source': source,
        'stratum': stratum,
        'description': description,
        'url': url,
        **kwargs
    }


# =============================================================================
# SKLEARN DATASETS
# =============================================================================

def _load_friedman1(n_samples: int = 2000, noise: float = 0.1,
                    n_noise_features: int = 0):
    """Load Friedman #1 dataset (smooth nonlinear)."""
    from sklearn.datasets import make_friedman1

    X, y = make_friedman1(n_samples=n_samples, n_features=5, noise=noise,
                          random_state=42)

    # Add noise features if requested
    if n_noise_features > 0:
        rng = np.random.RandomState(42)
        X_noise = rng.randn(n_samples, n_noise_features)
        X = np.hstack([X, X_noise])

    cols = [f'x{i}' for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols), y


def _load_friedman2(n_samples: int = 2000, noise: float = 0.1):
    """Load Friedman #2 dataset."""
    from sklearn.datasets import make_friedman2
    X, y = make_friedman2(n_samples=n_samples, noise=noise, random_state=42)
    cols = [f'x{i}' for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols), y


def _load_friedman3(n_samples: int = 2000, noise: float = 0.1):
    """Load Friedman #3 dataset."""
    from sklearn.datasets import make_friedman3
    X, y = make_friedman3(n_samples=n_samples, noise=noise, random_state=42)
    cols = [f'x{i}' for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols), y


def _load_california_housing():
    """Load California housing dataset."""
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=True)
    return data.data, data.target.values


def _load_diabetes():
    """Load diabetes dataset."""
    from sklearn.datasets import load_diabetes
    data = load_diabetes(as_frame=True)
    return data.data, data.target.values


# Register sklearn datasets
register_dataset(
    'friedman1', _load_friedman1, 'sklearn', 'S1',
    'Friedman #1: y = 10*sin(pi*x0*x1) + 20*(x2-0.5)^2 + 10*x3 + 5*x4',
    'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman1.html',
)
register_dataset(
    'friedman1_d100',
    lambda: _load_friedman1(n_samples=2000, n_noise_features=95),
    'sklearn', 'S1',
    'Friedman #1 with 95 noise features (d=100 total)',
)
# friedman2, friedman3 removed (24Jan26) - S1 over-represented (21→16 datasets)
# register_dataset(
#     'friedman2', _load_friedman2, 'sklearn', 'S1',
#     'Friedman #2: y = sqrt(x0^2 + (x1*x2 - 1/(x1*x3))^2)',
# )
# register_dataset(
#     'friedman3', _load_friedman3, 'sklearn', 'S1',
#     'Friedman #3: y = atan((x1*x2 - 1/(x1*x3))/x0)',
# )
register_dataset(
    'california_housing', _load_california_housing, 'sklearn', 'S4',
    'California housing prices (median house value)',
    'https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset',
)
register_dataset(
    'diabetes', _load_diabetes, 'sklearn', 'S3',
    'Diabetes progression prediction (life sciences)',
)


# =============================================================================
# OPENML DATASETS (Grinsztajn benchmark)
# =============================================================================

def _load_openml(data_id: int, target_col: str = None):
    """Load dataset from OpenML.

    Returns X with categorical columns preserved (as 'category' or 'object' dtype).
    The DatasetPreprocessor will handle encoding via TargetEncoder.
    """
    from sklearn.datasets import fetch_openml

    data = fetch_openml(data_id=data_id, as_frame=True, parser='auto')
    X = data.data.copy()  # Copy to avoid modifying cached data
    y = data.target

    # Handle target column specification
    if target_col and target_col in X.columns:
        y = X[target_col]
        X = X.drop(columns=[target_col])

    # Convert target to numeric
    y = pd.to_numeric(y, errors='coerce').values

    return X, y


# Grinsztajn numerical regression suite (OpenML 297)
# Strata: S1=Engineering, S2=Tabular, S3=Physics, S4=Threshold-heavy
OPENML_NUMERICAL = {
    # Engineering/Control (S1)
    'cpu_act': {'data_id': 44132, 'desc': 'CPU activity prediction', 'stratum': 'S1'},
    'pol': {'data_id': 44133, 'desc': 'Telecommunications satisfaction (0-100%)', 'stratum': 'S2', 'ordinal': True, 'discrete': True},
    'elevators': {'data_id': 44134, 'desc': 'Elevators control', 'stratum': 'S1'},
    'Ailerons': {'data_id': 44137, 'desc': 'Ailerons control', 'stratum': 'S1'},
    # Behavioral/Social (S2)
    'wine_quality': {'data_id': 44136, 'desc': 'Wine quality score', 'stratum': 'S2', 'ordinal': True, 'discrete': True},
    'Bike_Sharing_Demand': {'data_id': 44142, 'desc': 'Bike sharing demand (counts)', 'stratum': 'S2', 'discrete': True},
    # 'year' removed - OpenML data_id 4352 not found, not in HuggingFace benchmark either
    # Physical/Chemical/Life Sciences (S3)
    'sulfur': {'data_id': 44145, 'desc': 'Sulfur recovery chemistry', 'stratum': 'S3'},
    'superconduct': {'data_id': 44148, 'desc': 'Superconductor critical temp', 'stratum': 'S3'},
    # Economic/Pricing (S4)
    'nyc-taxi-green-dec-2016': {'data_id': 44143, 'desc': 'NYC taxi fares (pricing)', 'stratum': 'S4'},
    # yprop_4_1 removed - 99% zeros, not standard regression (21Jan26)
    # Pricing/threshold-driven (S4)
    # 'houses' removed - duplicate of california_housing (sklearn)
    'house_16H': {'data_id': 44139, 'desc': 'House prices 16H', 'stratum': 'S4'},
    'diamonds': {'data_id': 44140, 'desc': 'Diamond prices', 'stratum': 'S4'},
    'Brazilian_houses': {'data_id': 44141, 'desc': 'Brazilian house prices', 'stratum': 'S4'},
    'house_sales': {'data_id': 44144, 'desc': 'House sales prices', 'stratum': 'S4'},
    'medical_charges': {'data_id': 44146, 'desc': 'Medical charges (insurance)', 'stratum': 'S4'},
    'MiamiHousing2016': {'data_id': 44147, 'desc': 'Miami housing prices', 'stratum': 'S4'},
}

for name, info in OPENML_NUMERICAL.items():
    # Extract extra kwargs (like ordinal) to pass through
    extra_kwargs = {k: v for k, v in info.items() if k not in ['desc', 'stratum', 'data_id']}
    register_dataset(
        name,
        lambda data_id=info['data_id']: _load_openml(data_id),
        'openml', info.get('stratum', 'S2'),
        info['desc'],
        f"https://www.openml.org/d/{info['data_id']}",
        **extra_kwargs
    )


# Grinsztajn categorical regression suite (OpenML 299)
# Strata: S1=Engineering, S2=Tabular, S3=Physics, S4=Threshold-heavy
OPENML_CATEGORICAL = {
    # Physics/Scientific (S4)
    # 'topo_2_1' removed - target is categorical ('DOWN'/'UP'), not regression
    # 'visualizing_soil' removed - 40-class classification (soil types 1-40), not regression (21Jan26)
    #   Note: was also wrong data_id (44158=KDDCup09_upselling binary), correct is 44056
    'particulate-matter-ukair-2017': {'data_id': 42207, 'desc': 'UK PM10 air quality hourly', 'stratum': 'S3'},  # Fixed data_id (was 44162=COMPAS)
    # Engineering (S1)
    # 'SGEMM_GPU_kernel_performance' removed - OpenML data_id 44163 not found
    # 'Mercedes_Benz_Greener_Manufacturing' removed - binary classification (2 classes), not regression
    # General tabular (S2)
    'analcatdata_supreme': {'data_id': 504, 'desc': 'Supreme court decisions (log-counts)', 'stratum': 'S2', 'discrete': True},  # Log-transformed counts
    # 'black_friday' removed - binary classification (2 classes), not regression
    # Insurance/threshold (S4)
    # NOTE: ID 44160 was wrong ("rl" classification dataset). Fixed to 42571 (real Allstate, 188K rows)
    'Allstate_Claims_Severity': {'data_id': 42571, 'desc': 'Insurance claim severity (loss $)', 'stratum': 'S4'},
}

for name, info in OPENML_CATEGORICAL.items():
    # Extract extra kwargs (like ordinal) to pass through
    extra_kwargs = {k: v for k, v in info.items() if k not in ['desc', 'stratum', 'data_id']}
    register_dataset(
        name,
        lambda data_id=info['data_id']: _load_openml(data_id),
        'openml', info.get('stratum', 'S2'),
        info['desc'] + ' (has categoricals)',
        f"https://www.openml.org/d/{info['data_id']}",
        **extra_kwargs
    )


# =============================================================================
# UCI DATASETS (Phase 1 - using ucimlrepo)
# =============================================================================

def _load_uci(uci_id: int, target_col: str = None):
    """Load dataset from UCI ML Repository using ucimlrepo."""
    from ucimlrepo import fetch_ucirepo

    data = fetch_ucirepo(id=uci_id)
    X = data.data.features
    y = data.data.targets

    # Handle multiple targets - use first or specified
    if y.shape[1] > 1:
        if target_col and target_col in y.columns:
            y = y[target_col]
        else:
            y = y.iloc[:, 0]  # Use first target
    else:
        y = y.iloc[:, 0]

    return X, y.values


# Engineering/Applied (S1) - UCI IDs
UCI_ENGINEERING = {
    'concrete_strength': {
        'uci_id': 165,
        'stratum': 'S1',
        'desc': 'Concrete compressive strength (n=1030, d=8)',
    },
    'airfoil_noise': {
        'uci_id': 291,
        'stratum': 'S1',
        'desc': 'NASA airfoil self-noise (n=1503, d=5)',
    },
    'power_plant': {
        'uci_id': 294,
        'stratum': 'S1',
        'desc': 'Combined cycle power plant output (n=9568, d=4)',
    },
}

for name, info in UCI_ENGINEERING.items():
    register_dataset(
        name,
        lambda uci_id=info['uci_id']: _load_uci(uci_id),
        'uci', info['stratum'],
        info['desc'],
        f"https://archive.ics.uci.edu/dataset/{info['uci_id']}"
    )

# Energy efficiency - heating load only (cooling is 97.6% correlated, essentially duplicate)
def _load_energy_efficiency_heating():
    return _load_uci(242, target_col='Y1')

register_dataset(
    'energy_efficiency_heating', _load_energy_efficiency_heating, 'uci', 'S1',
    'Building heating load prediction (n=768, d=8)',
    'https://archive.ics.uci.edu/dataset/242'
)

# forest_fires removed - 48% zeros, not standard regression (21Jan26)

# Student Performance - S2 small (mixed survey/tabular data)
def _load_student_performance():
    X, y = _load_uci(320, target_col='G3')  # Final grade (0-19)
    return X, y

register_dataset(
    'student_performance', _load_student_performance, 'uci', 'S2',
    'Portuguese student final grades (n=649, d=30) - survey/mixed features [ORDINAL: 17 grade levels 0-19]',
    'https://archive.ics.uci.edu/dataset/320',
    ordinal=True, discrete=True
)

# =============================================================================
# KAGGLE DATASETS (faster downloads, same data as UCI)
# =============================================================================

def _load_kaggle(dataset_slug: str, csv_name: str, target_col: str,
                 drop_cols: list = None):
    """
    Load dataset from Kaggle.

    Requires: KAGGLE_API_TOKEN env var or ~/.kaggle/kaggle.json
    Install: micromamba install -c conda-forge kaggle

    Parameters
    ----------
    dataset_slug : str
        Kaggle dataset identifier (e.g., 'thehapyone/uci-online-news-popularity-data-set')
    csv_name : str
        Name of CSV file inside the downloaded archive
    target_col : str
        Name of target column
    drop_cols : list, optional
        Columns to drop (e.g., URLs, IDs)
    """
    import subprocess
    import zipfile

    cache_dir = CACHE_DIR / 'kaggle'
    cache_dir.mkdir(parents=True, exist_ok=True)

    csv_path = cache_dir / csv_name

    if not csv_path.exists():
        # Download from Kaggle
        zip_path = cache_dir / f"{dataset_slug.split('/')[-1]}.zip"
        if not zip_path.exists():
            cmd = ['kaggle', 'datasets', 'download', '-d', dataset_slug,
                   '-p', str(cache_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Kaggle download failed: {result.stderr}")

        # Unzip
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(cache_dir)

    df = pd.read_csv(csv_path)

    # Handle column names with leading spaces (UCI quirk)
    df.columns = df.columns.str.strip()

    # Extract target
    y = df[target_col].values

    # Drop specified columns
    cols_to_drop = [target_col]
    if drop_cols:
        cols_to_drop.extend(drop_cols)
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    return X, y


def _load_online_news_kaggle():
    """Load Online News Popularity from Kaggle (faster than UCI)."""
    return _load_kaggle(
        dataset_slug='thehapyone/uci-online-news-popularity-data-set',
        csv_name='OnlineNewsPopularity.csv',
        target_col='shares',
        drop_cols=['url']  # URL is not a feature
    )


# Additional UCI datasets (kept for datasets not on Kaggle)
def _load_power_grid():
    X, y = _load_uci(471, target_col='stab')  # Use continuous stability target
    return X, y

# online_news_popularity EXCLUDED (25Jan26) - No signal: R² ≈ 0.02 for all models
# Target (shares) is extremely skewed (median=1400, max=843300), essentially noise
# register_dataset(
#     'online_news_popularity', _load_online_news_kaggle, 'kaggle', 'S2',
#     'Online news article shares prediction (n=39644, d=58) [EXCLUDED: R²~0.02, no signal]',
#     'https://www.kaggle.com/datasets/thehapyone/uci-online-news-popularity-data-set'
# )
register_dataset(
    'power_grid_stability', _load_power_grid, 'uci', 'S4',  # Stability thresholds
    'Power grid stability margin prediction (n=10000, d=12)',
    'https://archive.ics.uci.edu/dataset/471'
)

# Yacht Hydrodynamics - REMOVED: n=308 < 500 minimum (20Jan26)
# register_dataset(
#     'yacht_hydrodynamics',
#     lambda: _load_openml(42370),
#     'openml', 'S1',
#     'Yacht residuary resistance prediction (n=308, d=6)',
#     'https://www.openml.org/d/42370'
# )


# =============================================================================
# HUGGINGFACE DATASETS (Grinsztajn benchmark - inria-soda/tabular-benchmark)
# =============================================================================

def _load_huggingface_tabular(config_name: str, target_col: str = None):
    """Load dataset from inria-soda/tabular-benchmark on HuggingFace."""
    from datasets import load_dataset

    ds = load_dataset('inria-soda/tabular-benchmark', config_name)
    df = ds['train'].to_pandas()

    # Target is typically the last column
    if target_col and target_col in df.columns:
        y = df[target_col].values
        X = df.drop(columns=[target_col])
    else:
        # Assume last column is target
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1]

    return X, y


# Missing Grinsztajn regression datasets (not already in our OpenML/PMLB sets)
HF_GRINSZTAJN = {
    'abalone': {
        'config': 'reg_num_abalone',
        'stratum': 'S3',  # Life sciences - biological growth
        'desc': 'Abalone age prediction (ring counts, n=4177, d=8)',
        'discrete': True,
    },
    # 'seattlecrime6' removed - discretized target (29 values, multiples of 61), not continuous regression
    # 'Airlines_DepDelay_1M' removed - R²<0.05 for all models, essentially unpredictable noise
    # 'delays_zurich_transport' removed - R²<0.04 for all models, essentially unpredictable noise
    # NOTE: sgemm_gpu, miami_housing, allstate_claims removed - duplicates of OpenML versions
    # (SGEMM_GPU_kernel_performance, MiamiHousing2016, Allstate_Claims_Severity)
}

for name, info in HF_GRINSZTAJN.items():
    # Extract extra kwargs (like discrete) to pass through
    extra_kwargs = {k: v for k, v in info.items() if k not in ['config', 'stratum', 'desc']}
    register_dataset(
        name,
        lambda cfg=info['config']: _load_huggingface_tabular(cfg),
        'huggingface', info['stratum'],
        info['desc'],
        f"https://huggingface.co/datasets/inria-soda/tabular-benchmark",
        **extra_kwargs
    )


# =============================================================================
# TABARENA ADDITIONAL DATASETS (OpenML IDs from TabArena curation)
# =============================================================================

TABARENA_ADDITIONAL = {
    # Physics/Scientific (S3)
    'qsar_fish_toxicity': {
        'data_id': 46954,
        'stratum': 'S3',  # Physics/Scientific
        'desc': 'QSAR fish toxicity prediction (n=907, d=7)',
    },
    'physiochemical_protein': {
        'data_id': 46949,
        'stratum': 'S3',  # Physics/Scientific
        'desc': 'Protein physicochemical properties (n=45730, d=10)',
    },
    # Physics/Scientific + [high-d] tag
    'qsar_tid_11': {
        'data_id': 46953,
        'stratum': 'S3',  # Physics/Scientific + [high-d]
        'desc': 'QSAR-TID-11 molecular descriptors (n=5742, d=1025)',
    },
    # Threshold-heavy (S4) - medical insurance has pricing tiers
    'healthcare_insurance': {
        'data_id': 46931,
        'stratum': 'S4',
        'desc': 'Healthcare insurance expenses (n=1338, d=7)',
    },
    # Tabular (S2)
    'food_delivery_time': {
        'data_id': 46928,
        'stratum': 'S2',
        'desc': 'Food delivery time prediction (n=45451, d=10)',
    },
    'fiat_500_price': {
        'data_id': 46907,
        'stratum': 'S4',  # Car pricing has threshold effects
        'desc': 'Used Fiat 500 car prices (n=1538, d=8)',
    },
}

for name, info in TABARENA_ADDITIONAL.items():
    register_dataset(
        name,
        lambda data_id=info['data_id']: _load_openml(data_id),
        'openml', info['stratum'],
        info['desc'],
        f"https://www.openml.org/d/{info['data_id']}"
    )


# =============================================================================
# PMLB DATASETS
# =============================================================================

def _load_pmlb(dataset_name: str):
    """Load dataset from PMLB."""
    import pmlb
    X, y = pmlb.fetch_data(dataset_name, return_X_y=True, local_cache_dir=str(CACHE_DIR / 'pmlb'))
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
    return X, y


# Selected PMLB datasets for symbolic regression heritage
# Strata assigned based on domain characteristics:
# S1 = Engineering/smooth, S2 = Tabular/mixed, S3 = Physics/scientific, S4 = Discontinuous/threshold
PMLB_DATASETS = {
    # General tabular (S2)
    # '1027_ESL' removed - n=488 < 500 minimum (20Jan26)
    '1028_SWD': {'desc': 'SWD dataset (ordinal ratings)', 'stratum': 'S2', 'ordinal': True, 'discrete': True},
    '1029_LEV': {'desc': 'LEV dataset (ordinal ratings)', 'stratum': 'S2', 'ordinal': True, 'discrete': True},
    '1030_ERA': {'desc': 'ERA dataset (ordinal ratings)', 'stratum': 'S2', 'ordinal': True, 'discrete': True},
    # '192_vineyard' removed - n=52 < 500 minimum (20Jan26)
    # '229_pwLinear' removed - n=200 < 500 minimum (20Jan26)
    # '294_satellite_image' removed - classification (6 soil types), not regression (20Jan26)
    '4544_GeographicalOriginalofMusic': {'desc': 'Music geographic origin (d=117)', 'stratum': 'S2'},  # S2 + [high-d]
    # '519_vinnie' removed - n=380 < 500 minimum (20Jan26)
    # '523_analcatdata_neavote' removed - n=100 < 500 minimum (20Jan26)
    # '556_analcatdata_apnea2' removed - n=475 < 500 minimum (20Jan26)
    # '557_analcatdata_apnea1' removed - n=475 < 500 minimum (20Jan26)
    # '560_bodyfat' removed - n=252 < 500 minimum (20Jan26)
    # Engineering/Control systems (S1)
    # '197_cpu_act' removed - duplicate of cpu_act (OpenML)
    # '201_pol' removed - duplicate of pol (OpenML), same data with more features (20Jan26)
    '215_2dplanes': {'desc': '2D planes synthetic', 'stratum': 'S1'},
    '225_puma8NH': {'desc': 'Puma robot arm', 'stratum': 'S1'},
    # '227_cpu_small' removed - duplicate of cpu_act (same 8192 samples, 12 of 21 features) (25Jan26)
    # '228_elusage' removed - n=55 < 500 minimum (20Jan26)
    # '230_machine_cpu' removed - n=209 < 500 minimum (20Jan26)
    # '344_mv' removed - undocumented, trivially solvable (R²=0.9999 all models), not discriminative (21Jan26)
    # '561_cpu' removed - n=209 < 500 minimum (20Jan26)
    # '564_fried' removed (24Jan26) - Friedman synthetic (OpenML 564), redundant with friedman1
    # Same formula: Y = 10*sin(π*X1*X2) + 20*(X3-0.5)² + 10*X4 + 5*X5 + noise (n=40768, d=10)
    # '573_cpu_act' removed - duplicate of cpu_act (OpenML)
    # Physics/Environmental (S3)
    # '210_cloud' removed - n=108 < 500 minimum (20Jan26)
    '503_wind': {'desc': 'Irish wind speed from 12 met stations', 'stratum': 'S3'},  # Meteorology
    '522_pm10': {'desc': 'PM10 pollution', 'stratum': 'S3'},  # Physics/Scientific
    # '527_pm10' removed - doesn't exist in PMLB (527 is analcatdata_election2000)
    '529_pollen': {'desc': 'Pollen count', 'stratum': 'S3'},  # Physics/Scientific
    # '542_pollution' removed - n=60 < 500 minimum (20Jan26)
    '547_no2': {'desc': 'NO2 pollution', 'stratum': 'S3'},  # Physics/Scientific
    # Pricing/threshold-driven (S4)
    # '195_auto_price' removed - deprecated in PMLB
    '218_house_8L': {'desc': 'House prices 8L', 'stratum': 'S4'},
    # '537_houses' removed - duplicate of california_housing (sklearn) and houses (OpenML)
    # '574_house_16H' removed - duplicate of house_16H (OpenML)
    # High-dimensional (S3)
    # '505_tecator' removed - n=240 < 500 minimum (20Jan26)
    # '588_fri_c4_1000_100' removed (24Jan26) - Friedman synthetic (OpenML 588), redundant with friedman1_d100
    # Same formula with collinear noise features (n=1000, d=100). From Friedman (1999) Stochastic Gradient Boosting
}

for name, info in PMLB_DATASETS.items():
    # Extract extra kwargs (like ordinal) to pass through
    extra_kwargs = {k: v for k, v in info.items() if k not in ['desc', 'stratum']}
    register_dataset(
        f'pmlb_{name}',
        lambda n=name: _load_pmlb(n),
        'pmlb', info.get('stratum', 'S2'),
        info['desc'],
        f'https://epistasislab.github.io/pmlb/profile/{name}.html',
        **extra_kwargs
    )


# =============================================================================
# CHEMISTRY DATASETS (MoleculeNet - public sources)
# =============================================================================

# Check if RDKit is available for molecular descriptor computation
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# Standard molecular descriptors computed from SMILES
# These are "featurization" (converting structure to numbers), not "feature engineering"
# Analogous to extracting day-of-week from timestamps - standard practice, not clever engineering
#
# Descriptor selection rationale:
# - Cover key physicochemical properties (size, polarity, lipophilicity)
# - Include Lipinski rule-of-5 properties (drug-likeness indicators)
# - Provide structural counts (rings, rotatable bonds, heteroatoms)
# - All are fast to compute and well-established in chemoinformatics
#
RDKIT_DESCRIPTOR_INFO = {
    # Size/mass
    'mw': 'Molecular weight (Da) - molecular size',
    'n_heavy_atoms': 'Heavy atom count - size proxy excluding hydrogens',

    # Polarity/solubility-related
    'tpsa': 'Topological polar surface area (Å²) - polarity, membrane permeability',
    'n_hbd': 'H-bond donors - solubility, protein binding',
    'n_hba': 'H-bond acceptors - solubility, protein binding',

    # Lipophilicity
    'logp': 'Crippen LogP - octanol/water partition, lipophilicity estimate',

    # Flexibility/rigidity
    'n_rotatable': 'Rotatable bonds - molecular flexibility',
    'fraction_csp3': 'Fraction of sp3 carbons - saturation, 3D character',

    # Ring systems
    'n_rings': 'Total ring count - cyclicity',
    'n_aromatic_rings': 'Aromatic ring count - aromaticity',

    # Heteroatom content
    'n_heteroatoms': 'Heteroatom count (N, O, S, etc.) - polarity contribution',

    # Electronic properties
    'n_valence_electrons': 'Total valence electrons - electronic character',
    'formal_charge': 'Net formal charge - ionic character',
    'n_radical_electrons': 'Radical electrons - reactivity (usually 0)',

    # Other
    'mol_refractivity': 'Molar refractivity - polarizability proxy',
    'complexity': 'Structural complexity (spiro + bridgehead atoms)',
}


def compute_rdkit_descriptors(smiles_list: list) -> pd.DataFrame:
    """
    Compute standard RDKit molecular descriptors from SMILES.

    This is "featurization" - converting molecular structure to numerical features.
    Not domain-specific feature engineering (that would be intensive features, etc.).

    Returns DataFrame with 16 standard descriptors covering:
    - Size: mw, n_heavy_atoms
    - Polarity: tpsa, n_hbd, n_hba
    - Lipophilicity: logp
    - Flexibility: n_rotatable, fraction_csp3
    - Ring systems: n_rings, n_aromatic_rings
    - Heteroatoms: n_heteroatoms
    - Electronic: n_valence_electrons, formal_charge, n_radical_electrons
    - Other: mol_refractivity, complexity

    Requires RDKit: conda install -c conda-forge rdkit
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit not available. Install with: conda install -c conda-forge rdkit")

    descriptor_names = list(RDKIT_DESCRIPTOR_INFO.keys())
    descriptors = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            descriptors.append({k: np.nan for k in descriptor_names})
            continue

        desc = {
            'mw': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'n_hbd': Lipinski.NumHDonors(mol),
            'n_hba': Lipinski.NumHAcceptors(mol),
            'n_rotatable': Lipinski.NumRotatableBonds(mol),
            'n_rings': rdMolDescriptors.CalcNumRings(mol),
            'n_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'n_heavy_atoms': Lipinski.HeavyAtomCount(mol),
            'n_heteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
            'fraction_csp3': rdMolDescriptors.CalcFractionCSP3(mol),
            'mol_refractivity': Descriptors.MolMR(mol),
            'n_valence_electrons': Descriptors.NumValenceElectrons(mol),
            'n_radical_electrons': Descriptors.NumRadicalElectrons(mol),
            'formal_charge': Chem.GetFormalCharge(mol),
            'complexity': rdMolDescriptors.CalcNumSpiroAtoms(mol) + rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        }
        descriptors.append(desc)

    return pd.DataFrame(descriptors)


def _load_moleculenet_csv(url: str, target_col: str, smiles_col: str = 'smiles',
                          drop_cols: list = None, compute_descriptors: bool = True):
    """Load MoleculeNet dataset from public CSV URL."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Create cache filename from URL
    cache_file = CACHE_DIR / f"moleculenet_{url.split('/')[-1]}"

    if cache_file.exists():
        df = pd.read_csv(cache_file)
    else:
        df = pd.read_csv(url)
        df.to_csv(cache_file, index=False)

    # Extract target first
    y = df[target_col].values

    # Check if we have features beyond SMILES and identifiers
    id_patterns = ['id', 'ID', 'name', 'Name', 'Compound', 'compound', 'iupac', 'IUPAC']
    non_feature_cols = [smiles_col, target_col]
    if drop_cols:
        non_feature_cols.extend(drop_cols)
    for col in df.columns:
        if any(pat in col for pat in id_patterns):
            non_feature_cols.append(col)

    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    # If no features, compute RDKit descriptors from SMILES
    if len(feature_cols) == 0 and smiles_col in df.columns:
        if compute_descriptors and RDKIT_AVAILABLE:
            X = compute_rdkit_descriptors(df[smiles_col].tolist())
        else:
            if not RDKIT_AVAILABLE:
                warnings.warn(f"No pre-computed features and RDKit not available. "
                              f"Install RDKit for molecular descriptors: conda install -c conda-forge rdkit")
            X = pd.DataFrame()  # Empty
    else:
        # Use existing features
        X = df[feature_cols].copy()

    # Drop rows with missing target
    valid_mask = ~np.isnan(y)
    if len(X) > 0:
        X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask]

    return X, y


def _load_esol():
    """Load ESOL (Delaney) solubility dataset."""
    # Public URL from MoleculeNet/DeepChem
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv'
    return _load_moleculenet_csv(url, target_col='measured log solubility in mols per litre')


def _load_freesolv():
    """Load FreeSolv hydration free energy dataset."""
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv'
    # Drop 'calc' column - it's a DFT prediction, not a molecular descriptor
    return _load_moleculenet_csv(url, target_col='expt', drop_cols=['calc'])


def _load_lipophilicity():
    """Load Lipophilicity (logD) dataset."""
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv'
    return _load_moleculenet_csv(url, target_col='exp')


def _load_qm7():
    """Load QM7 atomization energy dataset with RDKit descriptors."""
    # QM7 CSV only has SMILES + target, so we compute RDKit descriptors
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7.csv'
    return _load_moleculenet_csv(url, target_col='u0_atom')


def _load_qm9_sample():
    """Load QM9 sample (subset for tractability)."""
    # QM9 is large - we'll use a preprocessed subset
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv'
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / 'qm9.csv'

    try:
        if cache_file.exists():
            df = pd.read_csv(cache_file, nrows=10000)  # Sample
        else:
            df = pd.read_csv(url, nrows=10000)
            df.to_csv(cache_file, index=False)

        # Target: HOMO-LUMO gap or similar
        target_col = 'gap' if 'gap' in df.columns else 'u0'
        y = df[target_col].values
        X = df.drop(columns=[target_col])

        # Drop SMILES/mol columns
        drop_cols = [c for c in X.columns if any(s in c.lower() for s in ['smiles', 'mol', 'inchi'])]
        X = X.drop(columns=drop_cols, errors='ignore')

        # CRITICAL: If target is 'gap' (homo-lumo gap), remove homo and lumo from features
        # to prevent target leakage (gap = lumo - homo)
        if target_col == 'gap':
            leakage_cols = [c for c in X.columns if c.lower() in ['homo', 'lumo']]
            if leakage_cols:
                X = X.drop(columns=leakage_cols, errors='ignore')

        return X, y
    except Exception as e:
        warnings.warn(f"QM9 loading failed: {e}. Returning empty dataset.")
        return pd.DataFrame(), np.array([])


# Register chemistry datasets
# Note on featurization:
# - ESOL: Has pre-computed features in source CSV (MW, TPSA, HBD, etc.)
# - FreeSolv, Lipophilicity: SMILES only - we compute 16 RDKit descriptors
# - QM7, QM9: Have their own feature representations
#
register_dataset(
    'esol', _load_esol, 'moleculenet', 'S3',
    'ESOL: Aqueous solubility (Delaney, 1128 molecules) [pre-computed features]',
    'https://moleculenet.org/datasets-1',
)
# FreeSolv: RDKit descriptors computed from SMILES (16 features)
register_dataset(
    'freesolv', _load_freesolv, 'moleculenet', 'S3',  # Physics/Scientific (chemistry)
    'FreeSolv: Hydration free energy (642 molecules) [RDKit featurization]',
    'https://moleculenet.org/datasets-1'
)
# Lipophilicity: RDKit descriptors computed from SMILES (16 features)
register_dataset(
    'lipophilicity', _load_lipophilicity, 'moleculenet', 'S3',  # Physics/Scientific (chemistry)
    'Lipophilicity: Octanol/water distribution (4200 molecules) [RDKit featurization]',
    'https://moleculenet.org/datasets-1'
)
register_dataset(
    'qm7', _load_qm7, 'moleculenet', 'S3',  # Physics/Scientific (quantum chemistry)
    'QM7: Atomization energies (7165 molecules) [Coulomb matrix features]',
    'https://moleculenet.org/datasets-1'
)
# qm9_sample removed - target leakage (gap = lumo - homo, features included both) (21Jan26)


# =============================================================================
# SYNTHETIC DISCONTINUOUS (S4)
# =============================================================================

def _make_step_function(n_samples: int = 2000, n_features: int = 8, noise: float = 0.5):
    """Create synthetic step function dataset."""
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, n_features)

    # Step function: y = sum of indicator functions
    y = np.zeros(n_samples)
    y += 2.0 * (X[:, 0] > 0)
    y += 3.0 * (X[:, 1] > 0.5)
    y -= 1.5 * (X[:, 2] < -0.5)
    y += 1.0 * ((X[:, 0] > 0) & (X[:, 1] > 0))  # Interaction

    y += noise * rng.randn(n_samples)

    cols = [f'x{i}' for i in range(n_features)]
    return pd.DataFrame(X, columns=cols), y


def _make_piecewise_linear(n_samples: int = 2000, n_features: int = 5, noise: float = 0.3):
    """Create piecewise linear dataset (ReLU-like)."""
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, n_features)

    # Piecewise linear: y = sum of ReLU-like functions
    y = np.zeros(n_samples)
    y += 2.0 * np.maximum(0, X[:, 0])
    y += 1.5 * np.maximum(0, -X[:, 1])
    y += np.maximum(0, X[:, 2] - 0.5)
    y -= np.maximum(0, X[:, 0] + X[:, 1])

    y += noise * rng.randn(n_samples)

    cols = [f'x{i}' for i in range(n_features)]
    return pd.DataFrame(X, columns=cols), y


register_dataset(
    'synthetic_step', _make_step_function, 'synthetic', 'S4',
    'Synthetic step function (discontinuous)',
)
register_dataset(
    'synthetic_piecewise', _make_piecewise_linear, 'synthetic', 'S4',
    'Synthetic piecewise linear (ReLU-like)',
)


def _make_multi_threshold(n_samples: int = 750, n_features: int = 6, noise: float = 0.3):
    """Create synthetic multi-threshold dataset (S4 small).

    Multiple discontinuities across dimensions - trees should strongly outperform
    smooth models. DT R²~0.96 vs Ridge R²~0.57 in testing.
    """
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, n_features)

    # Target with multiple threshold-based rules
    y = np.zeros(n_samples)
    y += 3.0 * (X[:, 0] > 0)                      # Step at x0=0
    y += 2.0 * (X[:, 1] > 0.5)                    # Step at x1=0.5
    y += 1.5 * (X[:, 2] < -0.3)                   # Step at x2=-0.3
    y += 1.0 * ((X[:, 3] > -1) & (X[:, 3] < 1))  # Plateau region
    y += 0.5 * (X[:, 0] > 0) * (X[:, 1] > 0)     # Interaction term

    y += noise * rng.randn(n_samples)

    cols = [f'x{i}' for i in range(n_features)]
    return pd.DataFrame(X, columns=cols), y


register_dataset(
    'synthetic_multithreshold', _make_multi_threshold, 'synthetic', 'S4',
    'Synthetic multi-threshold (n=750, d=6) - S4 small',
)


# =============================================================================
# FEYNMAN PHYSICS EQUATIONS (SRSD-Feynman from HuggingFace)
# =============================================================================
# Source: yoshitomo-matsubara/srsd-feynman_hard on HuggingFace
# Origin: MIT AI Feynman Database - ground-truth physics equations
# All datasets: n=8000 train samples, exact functional form known
# Stratum: S1 (Simulation) - these are equation-generated, not measurements

def _load_feynman_equation(eq_name: str):
    """Load a specific Feynman equation dataset from HuggingFace."""
    from huggingface_hub import hf_hub_download

    # Download the train split for this equation
    train_path = hf_hub_download(
        'yoshitomo-matsubara/srsd-feynman_hard',
        f'train/{eq_name}.txt',
        repo_type='dataset'
    )

    # Parse whitespace-delimited file (last column is target)
    df = pd.read_csv(train_path, sep=r'\s+', header=None)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].values

    # Name columns
    X.columns = [f'x{i}' for i in range(X.shape[1])]

    return X, y


# Selected Feynman equations covering different physics domains and complexities
# Format: {name: {eq_name, desc, d (dimensions)}}
FEYNMAN_DATASETS = {
    # Simple/low-dimensional
    'feynman_gaussian': {
        'eq_name': 'feynman-i.6.20',
        'desc': 'Gaussian distribution: exp(-x²/2σ²)/σ√2π (d=2)',
        'd': 2,
    },
    # feynman_ideal_gas removed (24Jan26) - S1 over-represented (21→16 datasets)
    # 'feynman_ideal_gas': {
    #     'eq_name': 'feynman-i.39.22',
    #     'desc': 'Ideal gas law: nkT/V (d=3)',
    #     'd': 3,
    # },
    # Medium complexity - trigonometric
    'feynman_wave_interference': {
        'eq_name': 'feynman-i.29.16',
        'desc': 'Wave interference: √(A₁² + A₂² + 2A₁A₂cos(δ)) (d=4)',
        'd': 4,
    },
    # feynman_wave_superposition removed (24Jan26) - S1 over-represented (21→16 datasets)
    # 'feynman_wave_superposition': {
    #     'eq_name': 'feynman-i.37.4',
    #     'desc': 'Wave superposition: I₁ + I₂ + 2√(I₁I₂)cos(δ) (d=3)',
    #     'd': 3,
    # },
    # feynman_gravitation removed (24Jan26) - S1 over-represented (21→16 datasets)
    # 'feynman_gravitation': {
    #     'eq_name': 'feynman-i.9.18',
    #     'desc': 'Gravitational force: Gm₁m₂/r² with 3D positions (d=8)',
    #     'd': 8,
    # },
}

for name, info in FEYNMAN_DATASETS.items():
    register_dataset(
        name,
        lambda eq=info['eq_name']: _load_feynman_equation(eq),
        'huggingface',
        'S1',  # Simulation - equation-generated data
        info['desc'],
        'https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_hard'
    )


# =============================================================================
# PUBLIC API
# =============================================================================

def list_datasets(stratum: str = None, source: str = None) -> pd.DataFrame:
    """
    List available datasets with their properties.

    Parameters
    ----------
    stratum : str, optional
        Filter by stratum (S1-S4)
    source : str, optional
        Filter by source (sklearn, openml, pmlb, moleculenet, synthetic)

    Returns
    -------
    DataFrame with dataset names and properties
    """
    rows = []
    for name, info in DATASET_REGISTRY.items():
        if stratum and info['stratum'] != stratum:
            continue
        if source and info['source'] != source:
            continue
        rows.append({
            'name': name,
            'source': info['source'],
            'stratum': info['stratum'],
            'description': info['description'][:50] + '...' if len(info['description']) > 50 else info['description'],
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(['stratum', 'source', 'name']).reset_index(drop=True)
    return df


def get_dataset_info(name: str) -> dict:
    """
    Get dataset registry info without loading data.

    Parameters
    ----------
    name : str
        Dataset name

    Returns
    -------
    dict with source, stratum, description, url
    """
    if name not in DATASET_REGISTRY:
        available = list(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available[:10]}...")

    info = DATASET_REGISTRY[name].copy()
    del info['loader']  # Don't expose the loader function
    return info


def load_dataset(name: str, compute_meta: bool = True, **kwargs) -> tuple:
    """
    Load a dataset by name.

    Parameters
    ----------
    name : str
        Dataset name (use list_datasets() to see available)
    compute_meta : bool
        If True, compute and return rich metadata
    **kwargs
        Additional arguments passed to the loader (e.g., n_samples for synthetic)

    Returns
    -------
    X : DataFrame
        Feature matrix
    y : ndarray
        Target values
    meta : DatasetMeta (if compute_meta=True)
        Rich metadata about the dataset
    """
    if name not in DATASET_REGISTRY:
        available = list(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available[:10]}...")

    info = DATASET_REGISTRY[name]
    loader = info['loader']

    # Load data
    try:
        X, y = loader(**kwargs) if kwargs else loader()
    except Exception as e:
        raise RuntimeError(f"Failed to load {name}: {e}")

    # Ensure X is DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])

    # Ensure y is numpy array
    y = np.asarray(y).ravel()

    if compute_meta:
        meta = compute_metadata(
            X, y, name,
            source=info['source'],
            stratum=info['stratum'],
            description=info['description'],
            url=info.get('url', '')
        )
        return X, y, meta
    else:
        return X, y


def load_multiple(names: list, compute_meta: bool = True) -> dict:
    """
    Load multiple datasets.

    Parameters
    ----------
    names : list of str
        Dataset names to load
    compute_meta : bool
        If True, compute rich metadata

    Returns
    -------
    dict of {name: (X, y, meta)} or {name: (X, y)}
    """
    results = {}
    for name in names:
        try:
            results[name] = load_dataset(name, compute_meta=compute_meta)
            print(f"  Loaded {name}: {results[name][0].shape}", flush=True)
        except Exception as e:
            warnings.warn(f"Failed to load {name}: {e}")
    return results


def get_stratum_datasets(stratum: str) -> list:
    """
    Get list of dataset names for a stratum.

    Parameters
    ----------
    stratum : str
        Stratum identifier (S1-S4)
    """
    return [name for name, info in DATASET_REGISTRY.items()
            if info['stratum'] == stratum]


def get_ordinal_datasets() -> list:
    """
    Get list of ordinal regression dataset names.

    Ordinal datasets have discrete, ordered targets (e.g., ratings 1-5)
    that are treated as regression targets but have ordinal semantics.

    These should be reported separately from continuous regression
    results for clarity, as ordinal regression is a distinct task.
    """
    return [name for name, info in DATASET_REGISTRY.items()
            if info.get('ordinal', False)]


def is_ordinal_dataset(name: str) -> bool:
    """Check if a dataset is ordinal regression."""
    if name not in DATASET_REGISTRY:
        return False
    return DATASET_REGISTRY[name].get('ordinal', False)


def get_discrete_datasets() -> list:
    """
    Get list of discrete-target dataset names.

    Discrete datasets have count-based or integer targets (e.g., counts, ages,
    log-transformed counts). This is broader than 'ordinal' - includes any
    dataset where the target takes discrete values rather than continuous.

    Note: ordinal datasets are a subset of discrete datasets.
    Smoothness/roughness metrics may behave differently on discrete targets.
    """
    return [name for name, info in DATASET_REGISTRY.items()
            if info.get('discrete', False)]


def is_discrete_dataset(name: str) -> bool:
    """Check if a dataset has discrete targets (counts, integers, etc.)."""
    if name not in DATASET_REGISTRY:
        return False
    return DATASET_REGISTRY[name].get('discrete', False)


def get_benchmark_datasets() -> list:
    """Get list of all registered dataset names."""
    return list(DATASET_REGISTRY.keys())


def get_benchmark_datasets_by_size(
    n_min: int = 500,
    n_max: int = 10000,
    dn_ratio_max: float = 0.1,
) -> list:
    """
    Get benchmark datasets filtered by size criteria.

    Parameters
    ----------
    n_min : int, default=500
        Minimum number of samples.
    n_max : int, default=10000
        Maximum number of samples. Use 50000 for large-scale benchmarks.
    dn_ratio_max : float, default=0.1
        Maximum features/samples ratio.

    Returns
    -------
    list of str
        Dataset names meeting criteria, sorted by n_samples.

    Examples
    --------
    >>> datasets = get_benchmark_datasets_by_size(n_min=500, n_max=10000)
    >>> datasets = get_benchmark_datasets_by_size(n_min=500, n_max=50000)
    """
    # Use cache file for fast lookup
    cache_path = Path(__file__).parent / 'dataset_sizes_cache.csv'

    if cache_path.exists():
        df = pd.read_csv(cache_path)
        # Filter by criteria
        mask = (
            (df['n_samples'] >= n_min) &
            (df['n_samples'] <= n_max) &
            ((df['n_features'] / df['n_samples']) < dn_ratio_max)
        )
        filtered = df[mask].sort_values('n_samples')
        return filtered['dataset'].tolist()
    else:
        # Fallback: load metadata for each dataset (slower)
        results = []
        for name in DATASET_REGISTRY:
            try:
                _, _, meta = load_dataset(name)
                n, d = meta.n_samples, meta.n_features
                if n >= n_min and n <= n_max and (d / n) < dn_ratio_max:
                    results.append((name, n))
            except Exception:
                continue
        results.sort(key=lambda x: x[1])
        return [name for name, _ in results]


def load_dataset_subsampled(
    name: str,
    n_max: int = 10000,
    random_state: int = 42,
):
    """
    Load dataset with optional subsampling for large datasets.

    Useful for benchmarking on large datasets with controlled sample size,
    following Grinsztajn's "medium-sized regime" approach.

    Parameters
    ----------
    name : str
        Dataset name.
    n_max : int, default=10000
        Maximum samples. Datasets larger than this are subsampled.
    random_state : int, default=42
        Random seed for reproducible subsampling.

    Returns
    -------
    X : ndarray
        Features (possibly subsampled).
    y : ndarray
        Target (possibly subsampled).
    meta : DatasetMeta
        Metadata (n_samples reflects actual returned size).
    """
    X, y, meta = load_dataset(name)

    if len(y) > n_max:
        rng = np.random.RandomState(random_state)
        indices = rng.choice(len(y), size=n_max, replace=False)
        indices.sort()  # Preserve order
        X = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
        y = y[indices]
        # Update metadata
        meta.n_samples = n_max

    return X, y, meta


# =============================================================================
# STRESS-TEST DATASETS (Large scale, TabPFN excluded)
# =============================================================================

# Large-scale datasets for testing scalability
# TabPFN has 10K limit (cloud) / 50K limit (local), so excluded from stress tests
STRESS_TEST_DATASETS = [
    # 'delays_zurich_transport' removed - R²<0.04, unpredictable
    # 'Airlines_DepDelay_1M' removed - R²<0.05, unpredictable
    'nyc-taxi-green-dec-2016',  # 582K rows, 9 features - taxi fares
    # 'black_friday' removed - binary classification, not regression
    'medical_charges',          # 163K rows, 3 features - simple large
    'diamonds',                 # 54K rows, 6 features - diamond prices
]


def get_stress_test_datasets() -> list:
    """
    Get list of large-scale stress-test datasets.

    These datasets exceed TabPFN's limits (10K cloud / 50K local).
    Use for testing scalability of ERBF, Chebyshev vs tree ensembles.

    Returns
    -------
    list of dataset names (5 datasets, 54K to 5.4M samples)
    """
    return STRESS_TEST_DATASETS.copy()


# High-N stress test: n > 100K
STRESS_HIGH_N_DATASETS = [
    # 'delays_zurich_transport' removed - R²<0.04, unpredictable
    # 'Airlines_DepDelay_1M' removed - R²<0.05, unpredictable
    'nyc-taxi-green-dec-2016',  # 582K rows, 9 features
    'medical_charges',          # 163K rows, 3 features
]

# High-D stress test: d > 100 (requires feature selection for TabPFN)
STRESS_HIGH_D_DATASETS = [
    'qsar_tid_11',                          # 5.7K rows, 1024 features
    'pmlb_4544_GeographicalOriginalofMusic',  # 1K rows, 117 features
    'pmlb_588_fri_c4_1000_100',             # 1K rows, 100 features
    'friedman1_d100',                       # 2K rows, 100 features
]


def get_stress_high_n_datasets() -> list:
    """
    Get high-N stress test datasets (n > 100K).

    These are very large datasets for testing scalability.
    Consider using --no-tune mode and/or subsampling.

    Returns
    -------
    list of dataset names (4 datasets, 163K to 5.4M samples)
    """
    return STRESS_HIGH_N_DATASETS.copy()


def get_stress_high_d_datasets() -> list:
    """
    Get high-D stress test datasets (d > 100).

    These exceed TabPFN's 100-feature limit.
    Use k-best Spearman feature selection for fair comparison.

    Returns
    -------
    list of dataset names (4 datasets, 100-1024 features)
    """
    return STRESS_HIGH_D_DATASETS.copy()


# =============================================================================
# PARTIAL BENCHMARK DATASETS
# =============================================================================

# Stratified partial benchmark: ~20 datasets for faster iteration
# 4 datasets per stratum where possible
PARTIAL_BENCHMARK_DATASETS = {
    'S1_smooth': [
        'friedman1',        # Classic smooth synthetic
        'friedman2',        # sqrt/atan composition
        'friedman3',        # atan composition
    ],
    'S2_tabular': [
        'cpu_act',             # OpenML - CPU activity
        'wine_quality',        # OpenML - wine scores
        'abalone',             # HuggingFace - marine biology
    ],
    'S3_physics': [
        'esol',                # Solubility (chemistry)
        'freesolv',            # Hydration free energy
        'lipophilicity',       # LogD
        'qm7',                 # Atomization energy
        'sulfur',              # Sulfur recovery
    ],
    'S4_threshold': [
        'california_housing',  # Housing prices (threshold effects)
        'synthetic_step',      # Step function
        'synthetic_piecewise', # ReLU-like
        'diamonds',            # OpenML - diamond prices
        'houses',              # OpenML - house prices
    ],
    'S2_categorical': [
        # 'black_friday' removed - binary classification, not regression
        'Allstate_Claims_Severity',  # Insurance with categoricals
    ],
    'PMLB_extra': [
        'pmlb_560_bodyfat',    # Body fat - small, classic
        'pmlb_529_pollen',     # Pollen count
    ],
}


def get_partial_benchmark_datasets() -> list:
    """
    Get list of stratified partial benchmark datasets (~20 datasets).

    For faster iteration during development:
    - Covers all 4 strata
    - ~3-4 hours runtime (vs 12-18 hours for full)
    - Representative of full benchmark characteristics

    Not for final reporting - use get_benchmark_datasets() for that.
    """
    datasets = []
    for stratum_datasets in PARTIAL_BENCHMARK_DATASETS.values():
        datasets.extend(stratum_datasets)
    return datasets


def get_partial_benchmark_by_stratum() -> dict:
    """
    Get partial benchmark datasets organized by stratum.

    Returns
    -------
    dict of {stratum: [dataset_names]}
    """
    return PARTIAL_BENCHMARK_DATASETS.copy()


# =============================================================================
# SUMMARY UTILITIES
# =============================================================================

def summarize_metadata(meta: DatasetMeta) -> dict:
    """Create summary dict from metadata (for DataFrame creation)."""
    return {
        'name': meta.name,
        'source': meta.source,
        'stratum': meta.stratum,
        'n_samples': meta.n_samples,
        'n_features': meta.n_features,
        'n_numerical': meta.n_numerical,
        'n_categorical': meta.n_categorical,
        'max_cardinality': max(meta.categorical_cardinalities.values()) if meta.categorical_cardinalities else 0,
        'target_range': meta.target_max - meta.target_min,
        'target_std': meta.target_std,
        'missing_frac': meta.missing_fraction,
    }


def create_dataset_summary(names: list = None) -> pd.DataFrame:
    """
    Create summary DataFrame for multiple datasets.

    Parameters
    ----------
    names : list, optional
        Dataset names to summarize. If None, uses priority datasets.

    Returns
    -------
    DataFrame with dataset properties
    """
    if names is None:
        # Priority datasets for benchmarking
        names = [
            'friedman1', 'friedman2', 'friedman3',
            'california_housing', 'diabetes',
            'esol', 'freesolv', 'lipophilicity',
            'synthetic_step', 'synthetic_piecewise',
        ]

    rows = []
    for name in names:
        try:
            _, _, meta = load_dataset(name)
            rows.append(summarize_metadata(meta))
        except Exception as e:
            print(f"  Skipping {name}: {e}", flush=True)

    return pd.DataFrame(rows)


def regenerate_metadata_cache(output_path: str = None, verbose: bool = True) -> pd.DataFrame:
    """
    Regenerate dataset metadata cache from DATASET_REGISTRY.

    Loads each dataset in the registry and extracts metadata (n_samples, n_features,
    stratum, source, ordinal) into a CSV cache for fast lookups.

    Parameters
    ----------
    output_path : str, optional
        Path to save CSV. Default: benchmark/data/dataset_sizes_cache.csv
    verbose : bool
        Print progress

    Returns
    -------
    df : DataFrame
        Metadata for all datasets
    """
    if output_path is None:
        output_path = Path(__file__).parent / 'dataset_sizes_cache.csv'
    else:
        output_path = Path(output_path)

    records = []
    total = len(DATASET_REGISTRY)

    if verbose:
        print(f"Regenerating metadata cache for {total} datasets...")

    for i, (name, info) in enumerate(sorted(DATASET_REGISTRY.items()), 1):
        if verbose and i % 10 == 0:
            print(f"  [{i}/{total}] {name}", flush=True)

        try:
            X, y, meta = load_dataset(name)
            records.append({
                'dataset': name,
                'n_samples': meta.n_samples,
                'n_features': meta.n_features,
                'stratum': meta.stratum,
                'source': meta.source,
                'ordinal': info.get('ordinal', False),
                'discrete': info.get('discrete', False),
            })
        except Exception as e:
            if verbose:
                print(f"  FAILED {name}: {e}")

    df = pd.DataFrame(records).sort_values('n_samples')
    df.to_csv(output_path, index=False)

    if verbose:
        print(f"\nSaved {len(df)} datasets to {output_path}")
        print(f"Strata breakdown:")
        for stratum in sorted(df['stratum'].unique()):
            count = (df['stratum'] == stratum).sum()
            print(f"  {stratum} ({STRATA[stratum]['name']}): {count}")

    return df


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == '__main__':
    print("Available datasets:")
    print(list_datasets())
    print()

    print("\nLoading test datasets...")
    for name in ['friedman1', 'california_housing', 'esol']:
        try:
            X, y, meta = load_dataset(name)
            print(f"\n{name}:")
            print(f"  Shape: {X.shape}")
            print(f"  Target range: [{meta.target_min:.2f}, {meta.target_max:.2f}]")
            print(f"  Numerical: {meta.n_numerical}, Categorical: {meta.n_categorical}")
            if meta.categorical_cardinalities:
                print(f"  Cardinalities: {meta.categorical_cardinalities}")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
