from numbers import Number
from typing import Tuple, List

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, Booster
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import r2_score, matthews_corrcoef
from sklearn.model_selection import train_test_split


class Featurizer:
    """
    Molecule featurizer - uses RDKit's Morgan fingerprints
    """
    def __init__(self, radius: int = 2, n_bits: int = 1024):
        """
        Initialize the featurizer

        Args:
            radius (int): Morgan radius
            n_bits (int): Number of bits
        """
        self.radius = radius
        self.n_bits = n_bits

        self.fn = lambda mol: AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=self.radius, nBits=self.n_bits
        ).ToList()

    def featurize_smiles(self, smiles: str) -> List[int]:
        """
        Featurize a SMILES string
        """
        mol = Chem.MolFromSmiles(smiles)
        return self.featurize_mol(mol)

    def featurize_mol(self, mol: Chem.Mol) -> List[int]:
        """
        Featurize a molecule
        """
        return self.fn(mol)


class AutoQsar:
    """
    Wrapper around lightgbm's LGBMRegressor and LGBMClassifier
    """
    def __init__(self, mode: str = "regression", **featurizer_kwargs):
        """
        Initialize AutoQsar
        Args:
            mode (str): "regression" or "classification"
            **featurizer_kwargs: Arguments for autoqsar.Featurizer
        """
        self.mode = mode

        if mode == "regression":
            self.model = LGBMRegressor()
            self.metric_name = "r2"
            self.metric = r2_score
        elif mode == "classification":
            self.model = LGBMClassifier()
            self.metric_name = "mcc"
            self.metric = matthews_corrcoef
        else:
            raise ValueError('mode must be one of "regression" or "classification"')

        self.featurizer = Featurizer(**featurizer_kwargs)
        self.loaded_from_params = False

    def fit(self, X, y) -> None:
        self.model.fit(X, y)

    def predict(self, X) -> np.ndarray:
        if self.loaded_from_params and (self.mode == "classification"):
            return self.model.predict(X) >= 0.5
        else:
            return self.model.predict(X)

    def predict_smiles(self, smiles: str) -> Number:
        features = self.featurizer.featurize_smiles(smiles)
        return self.predict([features])[0]

    def evaluate(self, X, y_true) -> float:
        pred = self.predict(X)
        return self.metric(pred, y_true)

    def save(self, output_path: str):
        self.model.booster_.save_model(output_path)

    @classmethod
    def from_params(cls, input_path: str, mode: str = "regression"):
        instance = cls(mode)
        instance.model = Booster(model_file=input_path)
        instance.loaded_from_params = True
        return instance


def train_autoqsar_model(
    df: pd.DataFrame,
    output_path: str,
    mode: str = "regression",
    smiles_col: str = "canonical_smiles",
    act_col: str = "pchembl_value",
    random_state: int = 42,
    **featurizer_kwargs
) -> Tuple[str, float]:
    """
    Train a model from a Pandas DataFrame.
    Uses a random train/test split with a test size of 10%

    Args:
        df (pandas.DataFrame): Data
        output_path (str): Output CSV path
        mode (str): "regression" or "classification"
        smiles_col (str): SMILES column
        act_col (str): Activity/class labels column
        random_state (int): Random state
        **featurizer_kwargs: Arguments for autoqsar.Featurizer

    Returns:

    """
    autoqsar = AutoQsar(mode=mode, **featurizer_kwargs)

    df.dropna(subset=[smiles_col, act_col], inplace=True)
    X = np.array([autoqsar.featurizer.featurize_smiles(smi) for smi in df[smiles_col]])
    y = df[act_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=random_state, test_size=0.1
    )
    autoqsar.fit(X_train, y_train)
    metric = autoqsar.evaluate(X_test, y_test)

    # Retrain on the whole dataset
    autoqsar = AutoQsar(mode=mode, **featurizer_kwargs)
    autoqsar.fit(X_train, y_train)
    autoqsar.save(output_path)

    return autoqsar.metric_name, metric
