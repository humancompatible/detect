from typing import Tuple

import numpy as np
import pandas as pd
from folktables import ACSDataSource, BasicProblem

"""
You can find all explanation of resources here:
https://www.icpsr.umich.edu/web/DSDR/studies/25042/datasets/0002/variables/POBP?archive=dsdr

In "Search Variables" filed, write the code of uknown abbreviation.
Or you can use:
https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2018.pdf
"""

# Common folktables features that take continuous values
CONTINUOUS_FEATURES = ["AGEP", "PINCP", "WKHP", "JWMNP", "POVPIP"]

# Mapping from codes to names
FEATURE_NAMES = {
    "SEX": "Sex",
    "RAC1P": "Race",
    "AGEP": "Age",
    "MAR": "Marital status",
    "POBP": "Place of birth",
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
    "COW": "Class of worker",
    "SCHL": "Educational attainment",
    "OCCP": "Occupation recode",
    "RELP": "Relationship",
    "WKHP": "Usual hours worked per week past 12 months",
}

# Mapping from protected feature values to their interpretation - for visualization
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
    "SCHL": {
        1: "No school completed",
        2: "Nursery school to grade 4",
        3: "Grade 5 or grade 6",
        4: "Grade 7 or grade 8",
        5: "Grade 9",
        6: "Grade 10",
        7: "Grade 11",
        8: "Grade 12 no diploma",
        9: "High school graduate",
        10: "Some college, but less than 1 year",
        11: "One or more years of college, no degree",
        12: "Associate's degree",
        13: "Bachelor's degree",
        14: "Master's degree",
        15: "Professional school degree",
        16: "Doctorate degree",
    },
    "COW": {
        # Not implemented yet. Look at docstrings of this function
    },
    "OCCP": {
        # Not implemented yet. Look at docstrings of this function
    },
    "RELP": {
        # Not implemented yet. Look at docstrings of this function
    },
    "WKHP": {
        # Not implemented yet. Look at docstrings of this function
    },
    # "POVPIP": {}, Poverty ratio has numeric values
}

# Some suggested feature simplifications for more statistically useful binning
FEATURE_PROCESSING = {
    "POBP": lambda x: int(x) // 100,  # group by continents + US
    "OCCP": lambda x: int(x) // 100,
    "PUMA": lambda x: int(x) // 100,
    "POWPUMA": lambda x: int(x) // 1000,
}

# A special folktables problem containing all 14 protected attributes used in the MSD paper
ProtectedOnly = BasicProblem(
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
    ],
    target="ST",
    # for fertility and other non-applicable features
    postprocess=lambda x: np.nan_to_num(x, -1),
)


def load_state_data(
    state: str,
    problem_cls: BasicProblem = ProtectedOnly,
    survey_year: str = "2018",
    horizon: str = "1-Year",
    survey="person",
    data_root: str = "../data/folktables",
    download: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download (if needed), load, and project a single-state folktables dataset.

    Args:
        state: Two-letter USPS state code (e.g. "CA", "FL").
        problem_cls: A folktables Problem class (ACSIncome, ACSPublicCoverage, ...).
        survey_year: e.g. "2018".
        horizon: "1-Year" (default) or "5-Year".
        survey: "person" (default) or "household".
        data_root: path under which folktables will write its CSVs.
        download: if True, attempt `get_data(download=True)`, else only local.

    Returns:
        X:      DataFrame with the data.
        y:      DataFrame of the target.
    """
    ds = ACSDataSource(
        survey_year=survey_year,
        horizon=horizon,
        survey=survey,
        root_dir=data_root,
    )
    try:
        raw = ds.get_data(states=[state], download=download)
    except Exception as e:
        print("\n⚠️  Automatic download failed:")
        print(f"    {e!r}\n")
        print("→ Please manually download this file and unzip it under:")
        print(f"    {data_root}/{survey_year}/{horizon}/csv_p{state.lower()}.zip")
        print("\nYou can get it from:")
        print(
            f"https://www2.census.gov/programs-surveys/acs/data/pums/{survey_year}/{horizon}/\n"
        )
        raw = ds.get_data(states=[state], download=False)

    X, y, _ = problem_cls.df_to_pandas(raw)
    return X, y


# ────────── folktables state‐code mapping ──────────
# ACS "place of birth" recodes for states (POBP)
# https://www.icpsr.umich.edu/web/DSDR/studies/25042/datasets/0002/variables/POBP?archive=dsdr
STATE_POBP_CODE: dict[str, int] = {
    "AL": 1,  # Alabama
    "AK": 2,  # Alaska
    "AZ": 4,  # Arizona
    "AR": 5,  # Arkansas
    "CA": 6,  # California
    "CO": 8,  # Colorado
    "CT": 9,  # Connecticut
    "DE": 10,  # Delaware
    "DC": 11,  # District of Columbia
    "FL": 12,  # Florida
    "GA": 13,  # Georgia
    "HI": 15,  # Hawaii
    "ID": 16,  # Idaho
    "IL": 17,  # Illinois
    "IN": 18,  # Indiana
    "IA": 19,  # Iowa
    "KS": 20,  # Kansas
    "KY": 21,  # Kentucky
    "LA": 22,  # Louisiana
    "ME": 23,  # Maine
    "MD": 24,  # Maryland
    "MA": 25,  # Massachusetts
    "MI": 26,  # Michigan
    "MN": 27,  # Minnesota
    "MS": 28,  # Mississippi
    "MO": 29,  # Missouri
    "MT": 30,  # Montana
    "NE": 31,  # Nebraska
    "NV": 32,  # Nevada
    "NH": 33,  # New Hampshire
    "NJ": 34,  # New Jersey
    "NM": 35,  # New Mexico
    "NY": 36,  # New York
    "NC": 37,  # North Carolina
    "ND": 38,  # North Dakota
    "OH": 39,  # Ohio
    "OK": 40,  # Oklahoma
    "OR": 41,  # Oregon
    "PA": 42,  # Pennsylvania
    "RI": 44,  # Rhode Island
    "SC": 45,  # South Carolina
    "SD": 46,  # South Dakota
    "TN": 47,  # Tennessee
    "TX": 48,  # Texas
    "UT": 49,  # Utah
    "VT": 50,  # Vermont
    "VA": 51,  # Virginia
    "WA": 53,  # Washington
    "WV": 54,  # West Virginia
    "WI": 55,  # Wisconsin
}


def state_to_pobp_code(abbrev: str) -> int:
    """
    Turn a two-letter state code (e.g. 'CA') into the ACS POBP recode.
    Raises a KeyError if the state isn't in the map.
    """
    st = abbrev.strip().upper()
    try:
        return STATE_POBP_CODE[st]
    except KeyError:
        valid = ", ".join(sorted(STATE_POBP_CODE.keys()))
        raise KeyError(
            f"Unknown state abbreviation '{abbrev}'. Valid codes are: {valid}"
        )
