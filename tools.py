import glob as g
import shelve
from datetime import datetime
from functools import lru_cache
from typing import Dict
from typing import Tuple, Union, List, Any, Optional, Sequence

import pandas as pd
import requests
import tqdm as t
from chembl_webresource_client.new_client import new_client
from langchain.tools import tool
from langchain.tools.base import ToolException
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import inchi

from autoqsar import AutoQsar, train_autoqsar_model

activity = new_client.activity
shelf = shelve.open("models")


def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df:
        try:
            df[col] = pd.to_numeric(df[col])
        except (TypeError, ValueError):
            pass

    return df


def smiles2inchikey(smi: str) -> Union[str, None]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return inchi.MolToInchiKey(mol)


def aggregate(series: pd.Series) -> Union[str, float]:
    if pd.api.types.is_numeric_dtype(series):
        return series.mean()
    else:
        return ";".join(set(series.astype(str)))


class QueryDataInputs(BaseModel):
    target_chembl_id: str = Field(description="ChEMBL target ID")
    output_csv: str = Field(description="Output CSV path")
    standard_types: Sequence[str] = Field(
        description='ChEMBL standard types (default: ("Kd", "Ki", "IC50"))',
        default=("Kd", "Ki", "IC50"),
    )


@tool(args_schema=QueryDataInputs)
def query_target_data(
    target_chembl_id: str,
    output_csv: str,
    standard_types: Sequence[str] = ("Kd", "Ki", "IC50"),
) -> Dict[str, Dict[str, Any]]:
    """
    Query ChEMBL data package for a ChEMBL target ID.

    If you want to query for a protein name, first convert to
    uniprot accession, then to ChEMBL target ID, and then query.

    Always make sure that the ChEMBL target ID is correct,
    unless provided by the user.

    Unless otherwise stated, use a meaningful name for
    the output CSV file (e.g. JAK2.csv, or AuroraA.csv).

    Args:
        target_chembl_id (str): ChEMBL target ID
        output_csv (str): Output CSV file path
        standard_types (Tuple[str]): ChEMBL standard types
            (default: ("Kd", "Ki", "IC50"))

    Returns:
        stats (Dict[str, Dict[str, Any]]): Description of the numerical columns
            as dictionary
    ------------------------------------
    """

    activities = activity.filter(
        target_chembl_id=target_chembl_id, pchembl_value__isnull=False
    ).filter(standard_type__in=standard_types)

    data = pd.DataFrame(
        [
            act
            for act in t.tqdm(
                activities, desc=f"Querying ChEMBL for {target_chembl_id}"
            )
        ]
    )

    if data.shape[0] == 0:
        raise ToolException(f"No data found for {target_chembl_id}")

    columns_to_keep = [
        "molecule_chembl_id",
        "canonical_smiles",
        "pchembl_value",
        "assay_chembl_id",
    ]
    data = convert_to_numeric(data[columns_to_keep])
    data["inchi_key"] = data.canonical_smiles.apply(smiles2inchikey)
    data.dropna(subset=["canonical_smiles", "inchi_key", "pchembl_value"], inplace=True)

    data = data.groupby("inchi_key").agg(aggregate).reset_index()

    data.to_csv(output_csv, index=False)

    return data.describe().to_dict()


class ResolveUniprotInputs(BaseModel):
    protein_name: str = Field(description="Protein name (e.g.: JAK2)")


@tool(args_schema=ResolveUniprotInputs)
@lru_cache
def protein_name_to_accession(
    protein_name: str,
) -> Tuple[str, str, str]:
    """
    Resolve the name of a protein to its
    Uniprot ID, name and accession

    Args:
        protein_name (str): Protein name (e.g. JAK2)

    Returns:
        uniprot_id (str): Uniprot ID (e.g. JAK2_HUMAN)
        name (str): Uniprot name (e.g. Tyrosine-protein kinase JAK2)
        accession (str): Uniprot accession (e.g. O60674)
    ------------------------------------
    """
    organism = "Homo sapiens"
    query = (
        f"https://rest.uniprot.org/uniprotkb/search?&"
        f'query=(protein_name:"{protein_name}")%20AND%20(organism_name:"{organism}")'
    )
    req = requests.get(query)
    results = req.json()["results"]

    if len(results) == 0:
        raise ToolException("No results found")
    result = results[0]

    uniprot_id = result["uniProtkbId"]
    accession = result["primaryAccession"]
    if "recommendedName" in result["proteinDescription"]:
        name = result["proteinDescription"]["recommendedName"]["fullName"]["value"]
    else:
        name = result["proteinDescription"]["submissionNames"][0]["fullName"]["value"]

    return uniprot_id, name, accession


class ResolveChEMBLInputs(BaseModel):
    accession: str = Field(description="Uniprot accession")


@tool(args_schema=ResolveChEMBLInputs)
@lru_cache
def accession_to_chembl_target_id(accession: str) -> Tuple[str, str, str]:
    """
    Resolve a Uniprot accession to a ChEMBL target name
    and a ChEMBL target ID

    Args:
        accession (str): Uniprot accession (e.g. O60674)

    Returns:
        name (str): Target name
        target_chembl_id (str): Target ChEMBL ID
        accession (str): Uniprot accession from ChEMBL
    ------------------------------------
    """
    req = requests.get(
        f"https://www.ebi.ac.uk/chembl/api/data/target/search?q={accession}&format=json"
    )
    results = req.json()["targets"]

    if len(results) == 0:
        raise ToolException("No results found")

    result = results[0]

    name = result["pref_name"]
    accession_from_query = result["target_components"][0]["accession"]
    target_chembl_id = result["target_chembl_id"]

    # if accession.upper() != accession_from_query.upper():
    #     return "Query accession and accession retrieved from ChEMBL do not match"

    return name, target_chembl_id, accession_from_query


class TrainModelInputs(BaseModel):
    input_csv: str = Field(description="Input ChEMBL CSV data package")
    mode: str = Field(description="Mode of the model: 'regression' or 'classification'")
    model_name: str = Field(description="Name of the model")
    output_path: str = Field(
        description="Output model parameters (e.g. JAK2_regression.txt)"
    )
    activity_threshold: Optional[float] = Field(
        description="Activity threshold if in classification mode"
    )


@tool(args_schema=TrainModelInputs)
def train_model(
    input_csv: str,
    mode: str,
    model_name: str,
    output_path: str,
    activity_threshold: Optional[float] = None,
) -> Tuple[str, float]:
    """
    Train a model from a CSV ChEMBL data package and save the parameters to disk.

    You can check if a CSV data package exists with 'check_local_files'.

    A typical model name includes the target name end the model mode
    (e.g. JAK2_regression or JAK2_classification)

    Args:
        input_csv (str): Input ChEMBL CSV data package
        mode (str): Model mode ('regression' or 'classification')
        model_name (str): Model name
        output_path (str): Output model parameters (e.g. JAK2_regression.txt)
        activity threshold (Optional[float]): Threshold for classification mode.
            A typical value is 6

    Returns:
        metric_name (str): Name of the model metric (e.g. 'r2' or 'mcc')
        metric (float): Value of the metric
    ------------------------------------
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = pd.read_csv(input_csv)
    if mode == "classification":
        if activity_threshold is None:
            raise ToolException(
                "activity_threshold must be defined for a classification model"
            )

        df["pchembl_value"] = df.pchembl_value >= activity_threshold

    metric_name, metric = train_autoqsar_model(df, output_path=output_path, mode=mode)

    shelf[model_name] = {
        "params_path": output_path,
        "metric_name": metric_name,
        "mode": mode,
        "performance": metric,
        "timestamp": timestamp,
    }
    shelf.sync()
    return metric_name, metric


class PredictMoleculeInputs(BaseModel):
    smiles: str = Field(description="Molecule SMILES string")
    model_name: str = Field(description="Name of the model to use")


@tool(args_schema=PredictMoleculeInputs)
def predict_molecule(smiles: str, model_name: str) -> float:
    """
    Predicts the activity of a single molecule according to
    an available model. Use 'get_models' to retrieve
    the list of models, otherwise train a new one.

    Args:
        smiles (str): Molecule SMILES string
        model_name (str): Name of the model
    ------------------------------------
    """

    models = get_models_not_tool()
    if model_name not in models:
        raise ToolException("Model not available - train it with 'train_model'")

    model = AutoQsar.from_params(
        models[model_name]["params_path"], mode=models[model_name]["mode"]
    )
    return model.predict_smiles(smiles)


@tool
def get_models() -> Dict:
    """
    Get the list of available models.
    ------------------------------------
    """
    return dict(shelf)


def get_models_not_tool() -> Dict:
    """
    Get the list of available models.
    ------------------------------------
    """
    return dict(shelf)


@tool
def check_local_files() -> List[str]:
    """
    Get the list of local files.
    ------------------------------------
    """
    return sorted(g.glob("./*"))


check_local_files.handle_tool_error = True
get_models.handle_tool_error = True
protein_name_to_accession.handle_tool_error = True
accession_to_chembl_target_id.handle_tool_error = True
query_target_data.handle_tool_error = True
train_model.handle_tool_error = True
predict_molecule.handle_tool_error = True
