# myapp/utils/column_mappings.py

COLUMN_RENAME_MAP = {
    "Datetime": "datetime",
    "AEP_MW": "aep_mw",
    # add more mappings as needed
}

def get_rename_map():
    return COLUMN_RENAME_MAP
