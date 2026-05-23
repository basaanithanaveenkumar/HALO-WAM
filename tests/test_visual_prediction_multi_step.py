import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "Halo_VLA"))

from config.model_config import HaloVLMConfig
from models.dit_frame_prediction import VisualDiTPredictor, make_dit_config
from models.halo_vla import HaloVLM


def test_dit_config_fallbacks():
    class DummyConfig:
        emb_dim = 128
        patch_size = 8
        dit_depth = 4
        dit_num_heads = 4
        dit_mlp_ratio = 2.0
        dit_max_resolution = 64

    cfg = DummyConfig()
    dit_cfg = make_dit_config(cfg, 3)
    assert dit_cfg.dit_patch_size == 8
    assert dit_cfg.dit_hidden_size == 128
    assert dit_cfg.dit_depth == 4
    assert dit_cfg.dit_num_heads == 4
    assert dit_cfg.dit_mlp_ratio == 2.0
    assert dit_cfg.dit_max_resolution == 64
    print("test_dit_config_fallbacks passed.")


def test_visual_dit_predictor_rgb_only_default():
    cfg = HaloVLMConfig()
    cfg.dit_patch_size = 4
    cfg.dit_hidden_size = 64
    cfg.dit_depth = 2
    cfg.dit_num_heads = 2
    cfg.emb_dim = 64

    predictor = VisualDiTPredictor(cfg)

    # Create dummy inputs
    B, T, C, H, W = 2, 2, 3, 32, 32
    past_rgb = torch.randn(B, T, C, H, W)
    future_rgb = torch.randn(B, C, H, W)
    context_emb = torch.randn(B, cfg.emb_dim)

    # By default, compute_losses should only calculate RGB CFM loss
    losses = predictor.compute_losses(past_rgb, future_rgb, context_emb)
    assert "loss_rgb_cfm" in losses
    assert "loss_visual_total" in losses
    assert "loss_depth_cfm" not in losses
    assert "loss_flow_cfm" not in losses
    print("test_visual_dit_predictor_rgb_only_default passed.")


def test_halo_vlm_multi_step_prediction_loss():
    cfg = HaloVLMConfig()
    cfg.img_size = 32
    cfg.patch_size = 16
    cfg.dit_patch_size = 4
    cfg.dit_hidden_size = 64
    cfg.dit_depth = 2
    cfg.dit_num_heads = 2
    cfg.emb_dim = 64
    cfg.vit_num_layers = 1
    cfg.dec_num_layers = 2

    model = HaloVLM(cfg)

    B, N, C, H, W = 2, 5, 3, 32, 32
    images = torch.randn(B, N, C, H, W)
    visual_context_emb = torch.randn(B, cfg.emb_dim)

    # With N=5, it should calculate loss over min(3, 4) = 3 future frames
    loss = model.compute_visual_prediction_loss(images, visual_context_emb=visual_context_emb)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    print("test_halo_vlm_multi_step_prediction_loss passed.")


def test_halo_vlm_predict_visual_future_autoregressive():
    cfg = HaloVLMConfig()
    cfg.img_size = 32
    cfg.patch_size = 16
    cfg.dit_patch_size = 4
    cfg.dit_hidden_size = 64
    cfg.dit_depth = 2
    cfg.dit_num_heads = 2
    cfg.emb_dim = 64
    cfg.vit_num_layers = 1
    cfg.dec_num_layers = 2

    model = HaloVLM(cfg)

    B, T, C, H, W = 2, 2, 3, 32, 32
    past_rgb = torch.randn(B, T, C, H, W)
    visual_context_emb = torch.randn(B, cfg.emb_dim)

    # Autoregressively predict 3 frames
    preds = model.predict_visual_future(past_rgb, visual_context_emb=visual_context_emb, num_frames=3)
    assert "rgb" in preds
    assert "depth" not in preds
    assert "flow" not in preds
    assert preds["rgb"].shape == (B, 3, C, H, W)
    print("test_halo_vlm_predict_visual_future_autoregressive passed.")


if __name__ == "__main__":
    print("Running tests...")
    test_dit_config_fallbacks()
    test_visual_dit_predictor_rgb_only_default()
    test_halo_vlm_multi_step_prediction_loss()
    test_halo_vlm_predict_visual_future_autoregressive()
    print("All tests passed successfully!")
