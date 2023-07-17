# QSAR-LLM

An attempt at training and calling QSAR models using large language models.

The repository implements:

1. A wrapper around `lightgbm`'s `LGBMRegressor` and `LGBMClassifier` that allows to easily train simple regression and
   classification models
2. A set of `langchain` tools to instruct LLM to download ChEMBL data packages for biological targets, train models and
   predict the actiivty of molecules

Model parameters are saved as `lightgbm.Booster` params TXT files, and information on all trained models is stored
using [`shelve`](https://docs.python.org/3/library/shelve.html)

The tools and the agent have been set up to be able to handle most parsing and internal errors, helping the LLM correct
its queries

## Set up the environment

```pip install -r requirements.txt```

## Tools

| name                            | description                                         |
|---------------------------------|-----------------------------------------------------|
| `get_models`                    | List shelved models                                 |
| `protein_name_to_accession`     | Convert protein name to UniProt accession           |
| `accession_to_chembl_target_id` | Convert UniProt accession to ChEMBL target ID       |
| `query_target_data`             | Query activity data in ChEMBL for a given target ID |
| `train_model`                   | Train a regression/classification `lightgbm` model  |
| `predict_molecule`              | Predict the activity of a molecule given a model    |

## Notebook

Check the [Jupyter Notebook](qsar-llm.ipynb) for examples

## Example

The query

````python
text = """
Train a regression model to predict activity for JAK2 using a fresh ChEMBL data package for the target.
Make sure to resolve the protein name to the correct ChEMBL target ID"""
agent.run(text)
````

Returns

````text
> Entering new AgentExecutor chain...

Action:
```
{
  "action": "protein_name_to_accession",
  "action_input": {
    "protein_name": "JAK2"
  }
}
```

Observation: ('JAK2_HUMAN', 'Tyrosine-protein kinase JAK2', 'O60674')
Thought: I need to use the accession to get the ChEMBL target ID
Action:
```
{
  "action": "accession_to_chembl_target_id",
  "action_input": {
    "accession": "O60674"
  }
}
```

Observation: ('Tyrosine-protein kinase JAK2', 'CHEMBL2971', 'O60674')
Thought: I need to query the ChEMBL data package
Action:
```
{
  "action": "query_target_data",
  "action_input": {
    "target_chembl_id": "CHEMBL2971",
    "output_csv": "JAK2.csv"
  }
}
```

Querying ChEMBL for CHEMBL2971: 100%|███████████████████████████████████████████| 7454/7454 [00:00<00:00, 15249.65it/s]

Observation: {'pchembl_value': {'count': 6484.0, 'mean': 7.341097866394738, 'std': 1.219005643404945, 'min': 3.84, '25%': 6.48, '50%': 7.31, '75%': 8.14, 'max': 10.97}}
Thought: I need to train a model
Action:
```
{
  "action": "train_model",
  "action_input": {
    "input_csv": "JAK2.csv",
    "mode": "regression",
    "model_name": "JAK2_regression",
    "output_path": "JAK2_regression.txt"
  }
}
```

Observation: ('r2', 0.5840131903651216)
Thought: I have the final answer
Action:
```
{
  "action": "Final Answer",
  "action_input": "I have trained a regression model for JAK2 with an R2 score of 0.58."
}
```
> Finished chain.

'I have trained a regression model for JAK2 with an R2 score of 0.58.'
````