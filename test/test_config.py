from src.config import load_config


def test_train_config_has_required_keys():
    cfg = load_config("configs/train.yaml")
    for key in ("seed", "batch_size", "epochs", "learning_rate", "model_path"):
        assert key in cfg