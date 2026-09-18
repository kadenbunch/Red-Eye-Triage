"""Model/data smoke test on synthetic images (skipped if PyTorch is not installed)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from torch.utils.data import DataLoader  # noqa: E402

from redeye_triage import config as C  # noqa: E402
from redeye_triage.data import EyeDiseaseDataset, get_transforms  # noqa: E402
from redeye_triage.models import compute_class_weights, get_model, predict_probs, train_model  # noqa: E402


@pytest.mark.parametrize("name", ["efficientnet", "mobilenet", "resnet18"])
def test_forward_train_predict(name):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 255, (8, C.IMG_SIZE, C.IMG_SIZE, 3), dtype=np.uint8)
    labels = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    train_tf, eval_tf = get_transforms()
    device = torch.device("cpu")
    model, cfg = get_model(name, len(C.CLASSES), pretrained=False)
    loader = DataLoader(EyeDiseaseDataset(data, labels, train_tf), batch_size=4, shuffle=True)
    model, hist = train_model(model, loader, compute_class_weights(labels, 4, device), epochs=1, device=device)
    yt, yp = predict_probs(model, DataLoader(EyeDiseaseDataset(data, labels, eval_tf), batch_size=4), device=device)
    assert yp.shape == (8, 4) and np.allclose(yp.sum(1), 1, atol=1e-5)
    assert cfg["output_classes"] == 4
