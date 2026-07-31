import os
import pandas as pd

from predictor import manager
from config import MODEL_CONFIG
from 预测_GAT_dropout_残差_改进 import predict_new_smiles


# 常见SMILES列名
SMILES_COLUMN_CANDIDATES = [

    "SMILE",
    "smile",
    "Smile",
    "SMILES",
    "smiles",
    "Smiles",
    "canonical_smiles",
    "Canonical_SMILES",
    "Structure",
    "structure"

]


# def load_file(filepath):
#     """
#     自动读取 csv / xlsx / xls
#     """
#
#     if filepath.endswith(".csv"):
#
#         df = pd.read_csv(filepath)
#
#     elif filepath.endswith(".xlsx"):
#
#         df = pd.read_excel(filepath)
#
#     elif filepath.endswith(".xls"):
#
#         df = pd.read_excel(filepath)
#
#     else:
#
#         raise ValueError("Unsupported file format.")
#
#     return df

def load_file(filepath):

    suffix = os.path.splitext(filepath)[1].lower()

    if suffix == ".csv":

        return pd.read_csv(filepath)

    elif suffix in [".xlsx", ".xls"]:

        return pd.read_excel(filepath)

    raise ValueError("Unsupported file format.")



def detect_smiles_column(df):
    """
    自动寻找SMILES列
    """

    for col in df.columns:

        if col in SMILES_COLUMN_CANDIDATES:

            return col

    return None


def batch_predict_dataframe(df, target):

    smiles_col = detect_smiles_column(df)

    if smiles_col is None:

        raise ValueError("Cannot find SMILES column.")

    model = manager.get_model(target)

    scaler = manager.get_scaler(target)

    prediction = []

    for smiles in df[smiles_col]:

        try:

            value = predict_new_smiles(

                smiles,

                model=model,

                scaler=scaler

            )

            prediction.append(round(float(value),4))

        except Exception:

            # prediction.append("Invalid SMILES")
            prediction.append(None)

    column_name = MODEL_CONFIG[target]["title"]

    df[column_name] = prediction

    return df


def save_result(df, output_path):
    suffix = os.path.splitext(output_path)[1].lower()
    if suffix == '.csv':
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
    elif suffix in ['.xlsx', '.xls']:
        df.to_excel(output_path, index=False)
    else:
        raise ValueError("Unsupported output format. Use .csv, .xlsx, or .xls")
    return output_path


def batch_predict_file(

    input_path,

    output_path,

    target

):

    df = load_file(input_path)

    df = batch_predict_dataframe(

        df,

        target

    )

    save_result(

        df,

        output_path

    )

    return output_path








