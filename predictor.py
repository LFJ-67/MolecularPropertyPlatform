import torch
import joblib

from config import MODEL_CONFIG

from 预测_GAT_dropout_残差_改进 import StableGNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelManager:

    def __init__(self):

        self.models = {}

        self.scalers = {}

        self.load_all_models()


    def load_all_models(self):

        print("=" * 60)
        print("Loading models...")
        print("=" * 60)

        for name, info in MODEL_CONFIG.items():

            model = StableGNN(
                node_feat_dim=19,
                fp_size=2048,
                mol_feat_dim=18
            ).to(DEVICE)

            state_dict = torch.load(
                info["model_path"],
                map_location=DEVICE,
                weights_only=True
            )

            model.load_state_dict(state_dict)

            model.eval()

            scaler = joblib.load(info["scaler_path"])

            self.models[name] = model

            self.scalers[name] = scaler

            print(f"√ {name} loaded")

        print("=" * 60)


    def get_model(self, target):

        return self.models[target]


    def get_scaler(self, target):

        return self.scalers[target]


manager = ModelManager()