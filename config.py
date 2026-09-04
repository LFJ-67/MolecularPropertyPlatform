import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CONFIG = {

    "triplet": {
        "title": "三线态能量",
        "unit": "kcal/mol",
        "model_path": os.path.join(BASE_DIR, "models", "triplet", "gnn_model.pth"),
        "scaler_path": os.path.join(BASE_DIR, "models", "triplet", "scaler.pkl"),
    },

    "singlet": {
        "title": "单线态能量",
        "unit": "kcal/mol",
        "model_path": os.path.join(BASE_DIR, "models", "singlet", "gnn_model.pth"),
        "scaler_path": os.path.join(BASE_DIR, "models", "singlet", "scaler.pkl"),
    },

    "oxidation": {
        "title": "氧化电势",
        "unit": "eV",
        "model_path": os.path.join(BASE_DIR, "models", "oxidation", "gnn_model.pth"),
        "scaler_path": os.path.join(BASE_DIR, "models", "oxidation", "scaler.pkl"),
    },

    "reduction": {
        "title": "还原电势",
        "unit": "eV",
        "model_path": os.path.join(BASE_DIR, "models", "reduction", "gnn_model.pth"),
        "scaler_path": os.path.join(BASE_DIR, "models", "reduction", "scaler.pkl"),
    },

    "absorption": {
        "title": "激发波长",
        "unit": "nm",
        "model_path": os.path.join(BASE_DIR, "models", "absorption", "gnn_model.pth"),
        "scaler_path": os.path.join(BASE_DIR, "models", "absorption", "scaler.pkl"),
    }

}