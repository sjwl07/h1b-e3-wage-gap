from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "archive"
YEARS = list(range(2017, 2023))
RAW_CSV = {y: ARCHIVE / f"LCA_FY_{y}.csv" for y in YEARS}

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
REPORT = ROOT / "report"

ALL_PATH = DATA_RAW / "lca_all.parquet"
EXPLORE_PATH = DATA_RAW / "explore.parquet"
CONFIRM_PATH = DATA_RAW / "confirm.parquet"

SEED = 42
CONFIRM_FRACTION = 0.2

# Raw column -> harmonized name. FY2019 has Wage_Rate_Of_Pay (the offered wage)
# instead of Prevailing_Wage; see report/fy2019_wage_check.md.
RENAME = {
    "Visa_Class": "visa_class",
    "Employer_Name": "employer_name",
    "SOC_Title": "soc_title",
    "Job_Title": "job_title",
    "Full_Time_Position": "full_time",
    "Worksite": "worksite",
    "Prevailing_Wage": "prevailing_wage",
    "Wage_Rate_Of_Pay": "wage_offered_fy2019",
    "Unit_Of_Pay": "unit_of_pay",
    "Employer_Location": "employer_location",
    "Employer_Country": "employer_country",
    "Case_Status": "case_status",
    "Quarter": "quarter",
}

# FY2017-19 use upper-case labels, FY2020+ title case with spaces.
CASE_STATUS_MAP = {
    "CERTIFIED": "CERTIFIED",
    "CERTIFIED-WITHDRAWN": "CERTIFIED-WITHDRAWN",
    "WITHDRAWN": "WITHDRAWN",
    "DENIED": "DENIED",
    "Certified": "CERTIFIED",
    "Certified - Withdrawn": "CERTIFIED-WITHDRAWN",
    "Withdrawn": "WITHDRAWN",
    "Denied": "DENIED",
}

# Trailing legal-form tokens removed when building employer_key.
LEGAL_SUFFIXES = {
    "INC", "INCORPORATED", "LLC", "LC", "LLP", "LP", "LTD", "LIMITED", "CORP",
    "CORPORATION", "CO", "COMPANY", "PC", "PLLC", "PA", "PVT", "PRIVATE", "PLC",
    "GMBH", "AG", "SA", "NV", "BV",
}
