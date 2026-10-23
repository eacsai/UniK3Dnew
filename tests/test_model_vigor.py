"""Focused VIGOR wrapper checks; the real CPU model test is opt-in."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import hashlib
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from model_vigor import ModelVigor, target_size, UniK3D


class FakeUniK3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = 0

    def infer(self, rgb, *, camera, normalize):
        self.calls += 1
        assert rgb.shape == (2, 3, 320, 640)
        assert rgb.dtype == torch.uint8 and rgb.device.type == "cpu"
        assert normalize is True and camera is not None
        assert rgb[0, :, 250, 100].tolist() == [100, 100, 100]
        assert rgb[1, :, 250, 100].tolist() == [200, 200, 200]
        depth = torch.empty((2, 1, 320, 640), dtype=torch.float32)
        for b in range(2):
            depth[b].fill_(10.0 + b)
            depth[b, 0, 200, :6] = torch.tensor(
                [-1.0, 0.0, 80.0, np.nextafter(np.float32(80), np.float32(81)), 81.0, 160.0]
            )
        return {"distance": depth, "confidence": torch.ones_like(depth)}


class ModelVigorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inputs = []
        for b in range(2):
            image = np.full((320, 640, 3), 100 + 100 * b, dtype=np.uint8)
            image[:100] = 0
            image[150, 0] = [5, 5, 5]  # RGB sum15: excluded.
            image[150, 1] = [5, 5, 6]  # RGB sum16: retained.
            path = self.root / (str(b) + ".png")
            Image.fromarray(image).save(path)
            self.inputs.append(path)
        self.fake = FakeUniK3D()
        with patch("model_vigor.UniK3D.from_pretrained", return_value=self.fake):
            self.net = ModelVigor(SimpleNamespace(), torch.device("cpu"))
        self.assertEqual(self.fake.anchor.device.type, "cpu")
        self.assertEqual(self.fake.resolution_level, 9)
        self.assertFalse(self.fake.training)
        np.testing.assert_allclose(self.net.camera.params.cpu().numpy().reshape(-1),
                                   [500, 500, 320, 240, 1024, 512, 3.14159, 1.57079])

    def check_outputs(self, paths):
        self.assertEqual(len(list(self.root.rglob("*.npy"))), 2)
        for b, path in enumerate(paths):
            depth = np.load(path)
            self.assertEqual(depth.shape, (1, 320, 640))
            self.assertTrue(np.isfinite(depth).all())
            self.assertEqual(int(np.count_nonzero(depth[:, :100])), 0)
            self.assertEqual(depth[0, 150, 0], 0)
            self.assertEqual(depth[0, 150, 1], 10 + b)
            np.testing.assert_array_equal(depth[0, 200, :6], [0, 0, 80, 0, 0, 0])
            self.assertEqual(depth[0, 250, 100], 10 + b)
        self.assertEqual(self.fake.calls, 1)

    def test_list_batch_with_distinct_parent_directories(self):
        paths = [self.root / "a" / "first.npy", self.root / "b" / "nested" / "second.npy"]
        self.net.create_data(self.inputs, paths)
        self.check_outputs(paths)

    def test_tuple_batch_with_bare_filename(self):
        before = Path.cwd()
        try:
            os.chdir(self.root)
            paths = ("first.npy", "nested/second.npy")
            self.net.create_data(tuple(self.inputs), paths)
            self.check_outputs([self.root / p for p in paths])
        finally:
            os.chdir(before)

    def test_scalar_batches_fail_before_side_effects(self):
        dest = self.root / "must_not_exist" / "out.npy"
        for inputs, outputs in [(str(self.inputs[0]), [dest]),
                                (self.inputs, str(dest)), (self.inputs, dest),
                                (bytes(str(self.inputs[0]), "utf8"), [dest])]:
            with self.subTest(inputs=inputs, outputs=outputs):
                with self.assertRaises(TypeError):
                    self.net.create_data(inputs, outputs)
                self.assertFalse(dest.parent.exists())
        self.assertEqual(self.fake.calls, 0)

    def test_mismatched_lengths_fail_before_side_effects(self):
        dest = self.root / "must_not_exist" / "out.npy"
        with self.assertRaises(ValueError):
            self.net.create_data(self.inputs, [dest])
        self.assertFalse(dest.parent.exists())
        self.assertEqual(self.fake.calls, 0)


@unittest.skipUnless(os.environ.get("VIGOR_REAL_SMOKE") == "1", "opt-in real CPU model")
class RealModelVigorTests(unittest.TestCase):
    def test_real_cpu_batch_two(self):
        import cv2
        torch.set_num_threads(2)
        fixtures = json.loads(os.environ["VIGOR_REAL_INPUTS"])
        self.assertEqual(len(fixtures), 2)
        root = Path(os.environ["VIGOR_REAL_OUTPUT_DIR"])
        paths = [root / (f["city"] + ".npy") for f in fixtures]
        self.assertFalse(root.exists(), "real output directory must be new")
        protected = [Path(f[k]) for f in fixtures for k in ("input", "sky", "prior", "stored")]
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        before = {str(p): digest(p) for p in protected}
        snapshot = Path(os.environ["VIGOR_REAL_SNAPSHOT"]).resolve(strict=True)
        weights = snapshot / "model.safetensors"
        config = snapshot / "config.json"
        self.assertEqual(digest(weights), os.environ["VIGOR_REAL_WEIGHT_SHA256"])
        self.assertEqual(digest(config), os.environ["VIGOR_REAL_CONFIG_SHA256"])
        actual_loader = UniK3D.from_pretrained

        def load_snapshot(name):
            self.assertEqual(name, "lpiccinelli/unik3d-vitl")
            return actual_loader(str(snapshot), local_files_only=True)

        # Delegate to the real loader using the checked local snapshot, not mutable cache main.
        with patch("model_vigor.UniK3D.from_pretrained", side_effect=load_snapshot) as loaded:
            net = ModelVigor(SimpleNamespace(), torch.device("cpu"))
            loaded.assert_called_once_with("lpiccinelli/unik3d-vitl")
        net.train()  # Match the existing VIGOR runner, including child-module mode.
        self.assertEqual(net.model.device.type, "cpu")
        self.assertTrue(net.model.training)
        camera_before = net.camera.params.detach().cpu().clone().tolist()
        original_infer = net.model.infer
        captured = {}

        def record_infer(rgb, **kwargs):
            self.assertEqual(tuple(rgb.shape), (2, 3, 320, 640))
            outputs = original_infer(rgb, **kwargs)
            captured["distance"] = outputs["distance"].detach().cpu().clone()
            return outputs

        with patch.object(net.model, "infer", side_effect=record_infer) as called:
            net.create_data([f["input"] for f in fixtures], paths)
            self.assertEqual(called.call_count, 1)
        report = {"snapshot": str(snapshot), "resolved_weights": str(weights.resolve()),
                  "weight_sha256": digest(weights), "config_sha256": digest(config),
                  "device": str(net.model.device), "model_class": type(net.model).__name__,
                  "parameter_count": sum(p.numel() for p in net.model.parameters()),
                  "batch": 2, "infer_calls": called.call_count, "module_training": net.model.training,
                  "resolution_level": net.model.resolution_level,
                  "camera_before": camera_before,
                  "camera_after": net.camera.params.detach().cpu().tolist(), "samples": []}
        self.assertGreater(report["parameter_count"], 300_000_000)
        for b, (fixture, path) in enumerate(zip(fixtures, paths)):
            depth = np.load(path)
            self.assertEqual(depth.shape, (1, 320, 640))
            self.assertTrue(np.isfinite(depth).all())
            self.assertGreaterEqual(float(depth.min()), 0)
            self.assertLessEqual(float(depth.max()), 80)
            with Image.open(fixture["input"]) as image:
                rgb = torch.from_numpy(np.array(image.resize(target_size, Image.Resampling.BILINEAR))).permute(2, 0, 1)
            excluded = (rgb.sum(dim=0) < 16).numpy()
            self.assertEqual(int(np.count_nonzero(depth[0][excluded])), 0)
            expected = captured["distance"][b].clamp(min=0) * (~torch.from_numpy(excluded)).unsqueeze(0).float()
            expected[expected > 80] = 0
            np.testing.assert_array_equal(depth, expected.numpy())
            with Image.open(fixture["sky"]) as image:
                sky = np.array(image.resize(target_size, Image.Resampling.NEAREST)) >= 128
            interior = cv2.erode(sky.astype(np.uint8), np.ones((5, 5), np.uint8),
                                borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
            self.assertEqual(int(np.count_nonzero(depth[0][interior])), 0)
            prior = np.load(fixture["prior"]).copy()
            prior[prior > 80] = 0
            stored = np.load(fixture["stored"])
            report["samples"].append({"city": fixture["city"], "path": str(path), "sha256": digest(path),
                "shape": list(depth.shape), "min": float(depth.min()), "max": float(depth.max()),
                "raw_above80": int((captured["distance"][b] > 80).sum()),
                "rgb_excluded_pixels": int(excluded.sum()), "rgb_excluded_nonzero": int(np.count_nonzero(depth[0][excluded])),
                "png_sky_pixels": int(sky.sum()), "png_sky_nonzero": int(np.count_nonzero(depth[0][sky])),
                "interior_sky_pixels": int(interior.sum()), "interior_sky_nonzero": int(np.count_nonzero(depth[0][interior])),
                "mae_vs_prior_batch1_historical80_m": float(np.abs(depth.astype(np.float64)-prior).mean()),
                "mae_vs_stored_m": float(np.abs(depth.astype(np.float64)-stored).mean()),
                "equal_to_stored": bool(np.array_equal(depth, stored))})
        after = {str(p): digest(p) for p in protected}
        self.assertEqual(before, after)
        report["protected_sha256"] = after
        report["limitations"] = "One actual CPU batch2; camera state recorded, long multi-batch corpus and historical CUDA lineage not certified; PNG boundary not hard-masked."
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    unittest.main()
