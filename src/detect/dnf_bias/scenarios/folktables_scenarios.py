import logging

import numpy as np
import pandas as pd
from folktables import ACSDataSource

from binarizer import Binarizer
from data_handler import DataHandler

logger = logging.getLogger(__name__)

SCENARIOS = [
    "ACSIncome",
    "ACSPublicCoverage",
    "ACSMobility",
    "ACSEmployment",
    "ACSTravelTime",
    "DifferentStates-HI-ME",
    "DifferentStates-CA-WY",
    "DifferentStates-MS-NH",
    "DifferentStates-MD-MS",
    "DifferentStates-LA-UT",
]

# https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2018.pdf
# TODO check if N/A are filtered
PROTECTED_ATTRS = [
    "SEX",  # sex
    "RAC1P",  # race
    "AGEP",  # age
    # 'MAR', # marital status
    "POBP",  # place of birth
    "_POBP",  # simplified place of birth, custom
    "DIS",  # disability
    "CIT",  # citizenship
    "MIL",  # military service - 2 = veteran?
    "ANC",  # ancestry
    "NATIVITY",  # foreign or US native
    "DEAR",  # difficulty hearing
    "DEYE",  # difficulty seeing
    "DREM",  # cognitive difficulty
    "FER",  # gave birth recently
    "POVPIP",  # ratio of income to the poverty threshold
    # not in folktables
    # "DRATX",  # veteran service connected disability - binary
    # "VPS",  # Veteran period of service
]

# optionally simplify some features
FEATURE_PROCESSING = {
    "POBP": lambda x: int(x) // 100,  # 219 to 6
    "OCCP": lambda x: int(x) // 100,  # from ~500 to 100
    "PUMA": lambda x: int(x) // 100,  # from ~250 to 40
    "POWPUMA": lambda x: int(x) // 1000,  # from ~114 to 20
    # "PINCP": lambda x: np.sign(x) * np.log(np.abs(x)) if x != 0 else 0, # scaling for a narrower range
    # "SCHL": could be binned
    # "RELP": could be binned
}

FEATURE_NAMES = {
    "SEX": "Sex",
    "RAC1P": "Race",
    "AGEP": "Age",
    "MAR": "Marital status",
    "POBP": "Place of birth",
    "_POBP": "Place of birth",
    "DIS": "Disability",
    "CIT": "Citizenship",
    "MIL": "Military service",
    "ANC": "Ancestry",
    "NATIVITY": "Foreign or US native",
    "DEAR": "Difficulty hearing",
    "DEYE": "Difficulty seeing",
    "DREM": "Cognitive difficulty",
    "FER": "Gave birth last year",
    "POVPIP": "Income / Poverty threshold",
}

# not exact, shortened for succinctness
PROTECTED_VALUES_MAP = {
    "SEX": {1: "Male", 2: "Female"},
    "RAC1P": {
        1: "White",
        2: "Black",
        3: "American Indian",
        4: "Alaska Native",
        5: "Native tribes specified",
        6: "Asian",
        7: "Pacific Islander",
        8: "Some Other Race",
        9: "Two or More Races",
    },
    # "AGEP": {}, Age is not categorical
    "MAR": {
        1: "Married",
        2: "Widowed",
        3: "Divorced",
        4: "Separated",
        5: "Never married",
    },
    # "POBP": {}, Too many values, one for each country
    "_POBP": {
        0: "US",
        1: "Europe",
        2: "Asia",
        3: "Non-US Americas",
        4: "Africa",
        5: "Oceania",
    },
    "DIS": {1: "With a disability", 2: "Without a disability"},
    "CIT": {
        1: "Born in the US",
        2: "Born in external US teritories",
        3: "Born abroad of US parent(s)",
        4: "Naturalized US citizen",
        5: "Not a US citizen",
    },
    "MIL": {
        None: "N/A (<17 years)",
        1: "On active duty",
        2: "No more active duty",
        3: "Active duty for training",
        4: "Never served",
    },
    "ANC": {
        1: "Single",
        2: "Multiple",
        3: "Unclassified",
        4: "Not reported",
        8: "Hidden",
    },
    "NATIVITY": {1: "Native", 2: "Foreign born"},
    "DEAR": {1: "Yes", 2: "No"},
    "DEYE": {1: "Yes", 2: "No"},
    "DREM": {1: "Yes", 2: "No", None: "N/A (<5 years)"},
    "FER": {1: "Yes", 2: "No", None: "N/A"},
    # "POVPIP": {}, Poverty ratio has numeric values
}

CONTINUOUS_FEATURES = ["AGEP", "PINCP", "WKHP", "JWMNP", "POVPIP"]
# NONE for JWMNP should be checked if it happens


def all_protected_attributes_scenario(state_code):
    from folktables import BasicProblem

    def custom_filter(data):
        df = data.copy()
        return df

    scenario = BasicProblem(
        features=[
            "SEX",
            "RAC1P",
            "AGEP",
            "POBP",
            "DIS",
            "CIT",
            "MIL",
            "ANC",
            "NATIVITY",
            "DEAR",
            "DEYE",
            "DREM",
            "FER",
            "POVPIP", 
            "ST"
        ],
        target="ST",
        target_transform=lambda x: x == state_code,
        group="RAC1P",
        preprocess=custom_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    return scenario

def load_custom_scenarios(seed, n_max, year="2018", horizon="1-Year", **kwargs):
    Dataset = all_protected_attributes_scenario(kwargs["code_states"][0])

    data_source = ACSDataSource(survey_year=year, horizon=horizon, survey="person")
    data = data_source.get_data(states=kwargs["states"], download=True)
    input_data, target_data, _ = Dataset.df_to_pandas(data)

    mask = ~input_data.isnull().any(axis=1)
    logger.debug(f"Removing {input_data.shape[0] - mask.sum()} rows with nans")
    input_data = input_data[mask.values]
    target_data = target_data[mask.values]

    values = {}
    bounds = {}
    for col in input_data.columns:
        vals = input_data[col].unique()
        logger.debug(f"{col} has {vals.shape[0]} values")
        if col in FEATURE_PROCESSING:
            input_data[col] = input_data[col].map(FEATURE_PROCESSING[col])
            vals = input_data[col].unique()
            logger.debug(f"{col} changed - {vals.shape[0]} values")
        if vals.shape[0] <= 1:
            input_data.drop(columns=[col], inplace=True)
            continue
        if col not in CONTINUOUS_FEATURES:
            values[col] = vals
        else:
            bounds[col] = (min(vals), max(vals))

    # Splitting the dataset
    input_data_state_samples = {}
    target_data_state_samples = {}
    np.random.seed(seed)
    for state_type in kwargs["code_states"]:
        mask_state = input_data["ST"] == state_type

        input_data_state = input_data[mask_state].reset_index(drop=True)
        target_data_state = target_data[mask_state].reset_index(drop=True)

        n = input_data_state.shape[0]
        samples = np.random.choice(n, size=min(n_max, n) // 2, replace=False)

        input_data_state_samples[state_type] = input_data_state.iloc[samples]
        target_data_state_samples[state_type] = target_data_state[
            target_data_state.columns[0]
        ].iloc[samples]

    input_data = pd.concat(
        [
            input_data_state_samples[kwargs["code_states"][0]],
            input_data_state_samples[kwargs["code_states"][1]],
        ],
        axis=0,
        ignore_index=True,
    )
    target_data = pd.concat(
        [
            target_data_state_samples[kwargs["code_states"][0]],
            target_data_state_samples[kwargs["code_states"][1]],
        ],
        axis=0,
        ignore_index=True,
    )
    # Merged the two states into one dataset

    # print(input_data)

    dhandler = DataHandler.from_data(
        input_data,
        target_data,
        categ_map=values,
        bounds_map=bounds,
    )

    binarizer = Binarizer(dhandler, target_positive_vals=[True])

    protected_cols = [col for col in input_data.columns if col in PROTECTED_ATTRS]
    dhandler_protected = DataHandler.from_data(
        input_data[protected_cols],
        target_data,
        categ_map=values,
        bounds_map=bounds,
    )
    binarizer_protected = Binarizer(dhandler_protected, target_positive_vals=[True])

    return (
        binarizer,
        dhandler,
        input_data,
        target_data,
        binarizer_protected,
        input_data[protected_cols],
    )


def load_scenario(
    name, seed, n_max, state="CA", year="2018", horizon="1-Year", **kwargs
):
    if name == "ACSIncome":
        from folktables import ACSIncome as Dataset
    elif name == "ACSPublicCoverage":
        from folktables import ACSPublicCoverage as Dataset
    elif name == "ACSMobility":
        from folktables import ACSMobility as Dataset
    elif name == "ACSEmployment":
        from folktables import ACSEmployment as Dataset
    elif name == "ACSTravelTime":
        from folktables import ACSTravelTime as Dataset
    elif name.split("-")[0] == "DifferentStates":
        states = sorted([name.split("-")[1], name.split("-")[2]])
        if states == ["HI", "ME"]:  # Hawaii and Maine
            kwargs["states"] = ["HI", "ME"]
            kwargs["code_states"] = [15.0, 23.0]
            return load_custom_scenarios(seed, n_max, year, horizon, **kwargs)
        elif states == ["CA", "WY"]:  # California and Wyoming
            kwargs["states"] = ["CA", "WY"]
            kwargs["code_states"] = [6.0, 56.0]
        elif states == ["MS", "NH"]:  # Mississippi and New Hampshire
            kwargs["states"] = ["MS", "NH"]
            kwargs["code_states"] = [28.0, 33.0]
        elif states == ["MD", "MS"]:  # Maryland and Mississippi
            kwargs["states"] = ["MD", "MS"]
            kwargs["code_states"] = [24.0, 28.0]
        elif states == ["LA", "UT"]:  # Louisiana and Utah
            kwargs["states"] = ["LA", "UT"]
            kwargs["code_states"] = [22.0, 49.0]
        else:
            raise ValueError(f'Scenario "{name}" with given states does not exist.')
        
        return load_custom_scenarios(seed, n_max, year, horizon, **kwargs)
    else:
        raise ValueError(f'Scenario "{name}" does not exist.')

    data_source = ACSDataSource(survey_year=year, horizon=horizon, survey="person")
    states = [state] if state is not None else None
    data = data_source.get_data(states=states, download=True)
    input_data, target_data, _ = Dataset.df_to_pandas(data)

    mask = ~input_data.isnull().any(axis=1)
    logger.debug(f"Removing {input_data.shape[0] - mask.sum()} rows with nans")
    input_data = input_data[mask.values]
    target_data = target_data[mask.values]

    values = {}
    bounds = {}
    for col in input_data.columns:
        vals = input_data[col].unique()
        logger.debug(f"{col} has {vals.shape[0]} values")
        if col in FEATURE_PROCESSING:
            input_data[col] = input_data[col].map(FEATURE_PROCESSING[col])
            vals = input_data[col].unique()
            logger.debug(f"{col} changed - {vals.shape[0]} values")
        if vals.shape[0] <= 1:
            input_data.drop(columns=[col], inplace=True)
            continue
        if col not in CONTINUOUS_FEATURES:
            values[col] = vals
        else:
            bounds[col] = (min(vals), max(vals))
        
    # print(values)
    # print(bounds)

    np.random.seed(seed)
    n = input_data.shape[0]
    samples = np.random.choice(n, size=min(n_max, n), replace=False)

    input_data = input_data.iloc[samples]
    target_data = target_data[target_data.columns[0]].iloc[samples]
    dhandler = DataHandler.from_data(
        input_data,
        target_data,
        categ_map=values,
        bounds_map=bounds,
    )

    binarizer = Binarizer(dhandler, target_positive_vals=[True])

    protected_cols = [col for col in input_data.columns if col in PROTECTED_ATTRS]
    dhandler_protected = DataHandler.from_data(
        input_data[protected_cols],
        target_data,
        categ_map=values,
        bounds_map=bounds,
    )
    binarizer_protected = Binarizer(dhandler_protected, target_positive_vals=[True])

    return (
        binarizer,
        dhandler,
        input_data,
        target_data,
        binarizer_protected,
        input_data[protected_cols],
    )


def load_classif_scenario(name, seed, n_max, state="CA", year="2018", horizon="1-Year"):
    if name == "ACSIncome":
        from folktables import ACSIncome as Dataset
    elif name == "ACSPublicCoverage":
        from folktables import ACSPublicCoverage as Dataset
    elif name == "ACSMobility":
        from folktables import ACSMobility as Dataset
    elif name == "ACSEmployment":
        from folktables import ACSEmployment as Dataset
    elif name == "ACSTravelTime":
        from folktables import ACSTravelTime as Dataset
    else:
        raise ValueError(f'Scenario "{name}" does not exist.')

    data_source = ACSDataSource(survey_year=year, horizon=horizon, survey="person")
    states = [state] if state is not None else None
    data = data_source.get_data(states=states, download=True)
    input_data, target_data, _ = Dataset.df_to_pandas(data)
    mask = ~input_data.isnull().any(axis=1)
    logger.debug(f"Removing {input_data.shape[0] - mask.sum()} rows with nans")
    input_data = input_data[mask.values]
    target_data = target_data[mask.values]

    values = {}
    bounds = {}
    for col in input_data.columns:
        vals = input_data[col].unique()
        logger.debug(f"{col} has {vals.shape[0]} values")
        if col in FEATURE_PROCESSING:
            input_data[col] = input_data[col].map(FEATURE_PROCESSING[col])
            vals = input_data[col].unique()
            logger.debug(f"{col} changed - {vals.shape[0]} values")
        if vals.shape[0] <= 1:
            input_data.drop(columns=[col], inplace=True)
            continue
        if col not in CONTINUOUS_FEATURES:
            values[col] = vals
        else:
            bounds[col] = (min(vals), max(vals))

    # subsample
    np.random.seed(seed)
    n = input_data.shape[0]
    samples = np.random.choice(n, size=min(n_max, n), replace=False)

    input_data = input_data.iloc[samples]
    target_data = target_data[target_data.columns[0]].iloc[samples]
    dhandler = DataHandler.from_data(
        input_data,
        target_data,
        categ_map=values,
        bounds_map=bounds,
    )

    binarizer = Binarizer(dhandler, target_positive_vals=[True])

    protected_cols = [col for col in input_data.columns if col in PROTECTED_ATTRS]
    dhandler_protected = DataHandler.from_data(
        input_data[protected_cols],
        target_data,
        categ_map=values,
        bounds_map=bounds,
    )
    binarizer_protected = Binarizer(dhandler_protected, target_positive_vals=[True])

    return (
        binarizer,
        dhandler,
        input_data,
        target_data,
        binarizer_protected,
        input_data[protected_cols],
    )
