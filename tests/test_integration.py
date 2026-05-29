"""End-to-end integration test: train, export, and run inference."""

import torch

from vad.model import Resnet1D
from vad.train import export_model, train


def test_train_export_inference(
    tmp_path, dummy_dataset, model_config, train_params, expected_output_shape
):
    """Train a small model, export it, reload and verify inference output shape."""
    model_dir = tmp_path / "model_output"

    last_ckpt = train(
        data_dir=dummy_dataset,
        model_dir=model_dir,
        config=model_config,
        epochs=train_params["epochs"],
        batch_size=train_params["batch_size"],
        lr=train_params["lr"],
    )

    checkpoints = list(model_dir.glob("checkpoint_epoch_*.pt"))
    assert len(checkpoints) == train_params["epochs"]
    assert last_ckpt.exists()

    export_dir = tmp_path / "exported"
    export_model(
        checkpoint=last_ckpt,
        out_dir=export_dir,
        config=model_config,
        torchscript=True,
    )

    assert (export_dir / "model_state_dict.pt").exists()
    assert (export_dir / "model_scripted.pt").exists()

    model = Resnet1D(model_config)
    state_dict = torch.load(
        export_dir / "model_state_dict.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()

    batch = expected_output_shape[0]
    dummy_input = torch.randn(batch, model_config.in_channels, 65)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == expected_output_shape
    assert torch.isfinite(output).all()


def test_torchscript_inference_matches_eager(tmp_path, dummy_dataset, model_config, train_params):
    """Verify the TorchScript model matches the eager model output."""
    model_dir = tmp_path / "model_output"

    last_ckpt = train(
        data_dir=dummy_dataset,
        model_dir=model_dir,
        config=model_config,
        epochs=train_params["epochs"],
        batch_size=train_params["batch_size"],
        lr=train_params["lr"],
    )

    export_dir = tmp_path / "exported"
    export_model(
        checkpoint=last_ckpt,
        out_dir=export_dir,
        config=model_config,
        torchscript=True,
    )

    eager_model = Resnet1D(model_config)
    state_dict = torch.load(
        export_dir / "model_state_dict.pt", map_location="cpu", weights_only=True
    )
    eager_model.load_state_dict(state_dict)
    eager_model.eval()

    scripted_model = torch.jit.load(str(export_dir / "model_scripted.pt"), map_location="cpu")

    torch.manual_seed(42)
    test_input = torch.randn(2, model_config.in_channels, 65)
    with torch.no_grad():
        eager_output = eager_model(test_input)
        scripted_output = scripted_model(test_input)

    assert torch.allclose(eager_output, scripted_output, atol=1e-5)
