import pandas as pd

def load_and_clean_metrica_tracking(filepath):
    """Loads and cleans tracking data from Metrica."""
    df = pd.read_csv(filepath, skiprows=2, low_memory=False)
    new_cols = []
    for col in df.columns[:3]: new_cols.append(col.strip())
    current_name = None
    for col in df.columns[3:]:
        col = col.strip()
        if 'Unnamed' not in col and col != '':
            current_name = col; new_cols.append(f"{current_name}_x")
        else:
            new_cols.append(f"{current_name}_y")
    df.columns = new_cols
    PITCH_LENGTH, PITCH_WIDTH = 1, 1
    coord_cols = [col for col in df.columns if '_x' in col or '_y' in col]
    df[coord_cols] = df[coord_cols].apply(pd.to_numeric, errors='coerce')
    for col in df.columns:
        if '_x' in col: df[col] = df[col] * PITCH_LENGTH
        elif '_y' in col: df[col] = df[col] * PITCH_WIDTH
    return df

def load_match(filepath):
    df = pd.read_csv(filepath)

    return df