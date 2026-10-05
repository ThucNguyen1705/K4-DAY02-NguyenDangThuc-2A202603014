"""Kiểm tra tự viết cho code/ (chạy được trên CPU):
    python -m unittest test_own -v      (đứng trong thư mục code/)

Gồm: focal gamma=0 bằng CE, label smoothing eps=0 bằng CE và khớp PyTorch, CutMix trộn cả ảnh
lẫn nhãn và lam đúng diện tích, Mixup, trọng số lớp, 3 nhóm tham số, lịch LR, đóng băng giữ BN
ở eval, gộp BN sai số < 1e-5, temperature scaling tìm lại được T, gộp view, parse_overrides.
"""
import math
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import inference
import losses
import model as M
import train


class TestLosses(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.logits = torch.randn(64, 9, generator=g) * 3
        self.y = torch.randint(0, 9, (64,), generator=g)

    def test_focal_gamma0_equals_ce(self):
        fl = losses.FocalLoss(gamma=0.0)(self.logits, self.y)
        ce = F.cross_entropy(self.logits, self.y)
        self.assertLess(abs(fl.item() - ce.item()), 1e-6)

    def test_focal_downweights_easy(self):
        self.assertLess(losses.FocalLoss(2.0)(self.logits, self.y).item(),
                        F.cross_entropy(self.logits, self.y).item())

    def test_ls_eps0_equals_ce_and_matches_torch(self):
        self.assertLess(abs(losses.LabelSmoothingCE(0.0)(self.logits, self.y).item()
                            - F.cross_entropy(self.logits, self.y).item()), 1e-6)
        ref = F.cross_entropy(self.logits, self.y, label_smoothing=0.1)
        self.assertLess(abs(losses.LabelSmoothingCE(0.1)(self.logits, self.y).item() - ref.item()), 1e-6)

    def test_class_weights(self):
        counts = [630, 640, 620, 610, 640, 600, 640, 610, 5460]
        w = losses.class_weights(counts, 0.0)
        self.assertAlmostEqual(w.mean().item(), 1.0, places=5)
        self.assertLess(w[8].item(), w[0].item())
        wb = losses.class_weights(counts, 0.999)
        self.assertAlmostEqual(wb.sum().item(), 9.0, places=4)
        self.assertLess(wb[8].item(), wb[0].item())

    def test_cutmix_mixes_images_and_labels(self):
        np.random.seed(1)
        torch.manual_seed(1)
        x = torch.arange(8, dtype=torch.float32).view(8, 1, 1, 1).expand(8, 3, 32, 32).clone()
        y = torch.arange(8)
        for _ in range(20):
            xm, (ya, yb, lam) = losses.mix_batch(x, y, 1.0, "cutmix")
            self.assertTrue(torch.equal(ya, y))
            # mỗi ảnh i chỉ chứa giá trị của i (phần giữ lại) và của yb[i] (hộp dán vào)
            for i in range(8):
                own = (xm[i, 0] == float(y[i])).float().mean().item()
                if yb[i] != y[i]:
                    self.assertAlmostEqual(own, lam, places=6)  # lam = tỉ lệ diện tích thật còn lại
                    self.assertTrue(set(xm[i].unique().tolist()) <= {float(y[i]), float(yb[i])})
            self.assertGreaterEqual(lam, 0.0)
            self.assertLessEqual(lam, 1.0)

    def test_mixup_and_mixed_loss(self):
        np.random.seed(0)
        x = torch.randn(16, 3, 8, 8)
        y = torch.randint(0, 9, (16,))
        xm, (ya, yb, lam) = losses.mix_batch(x, y, 0.4, "mixup")
        perm_x = (xm - lam * x) / (1 - lam)
        self.assertTrue(torch.allclose(perm_x.sum(), x.sum(), atol=1e-3))
        crit = nn.CrossEntropyLoss()
        l = losses.mixed_loss(crit, self.logits[:16], (ya, yb, lam))
        ref = lam * crit(self.logits[:16], ya) + (1 - lam) * crit(self.logits[:16], yb)
        self.assertAlmostEqual(l.item(), ref.item(), places=6)


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.net = M.build_model("resnet18", pretrained=False, init="finetune")

    def test_param_groups(self):
        groups = M.param_groups(self.net, 1e-4, 1e-3, 0.05)
        by = {g["name"]: g for g in groups}
        self.assertEqual(set(by), {"backbone_decay", "backbone_no_decay", "head"})
        self.assertEqual(by["backbone_no_decay"]["weight_decay"], 0.0)
        self.assertTrue(all(p.ndim <= 1 for p in by["backbone_no_decay"]["params"]))
        self.assertTrue(all(p.ndim > 1 for p in by["backbone_decay"]["params"]))
        self.assertEqual(by["head"]["lr"], 1e-3)
        n = sum(len(g["params"]) for g in groups)
        self.assertEqual(n, len(list(self.net.parameters())))

    def test_frozen_keeps_bn_eval(self):
        net = M.build_model("resnet18", pretrained=False, init="frozen")
        groups = M.param_groups(net, 1e-4, 1e-3, 0.05)
        self.assertEqual([g["name"] for g in groups], ["head"])
        M.set_train_mode(net)
        self.assertFalse(net.bn1.training)
        self.assertTrue(net.get_classifier().training)
        before = net.bn1.running_mean.clone()
        net(torch.randn(4, 3, 64, 64))
        self.assertTrue(torch.equal(before, net.bn1.running_mean))

    def test_counts(self):
        self.assertAlmostEqual(M.count_params(self.net), 11.18, delta=0.05)
        g = M.count_gmacs(self.net, 224)
        self.assertGreater(g, 1.6)
        self.assertLess(g, 1.95)


class TestSchedule(unittest.TestCase):
    def test_warmup_cosine(self):
        total, warm = 100, 10
        f = [train.lr_factor(s, total, warm) for s in range(total)]
        self.assertAlmostEqual(f[0], 0.1)
        self.assertAlmostEqual(f[9], 1.0)
        self.assertTrue(all(a >= b for a, b in zip(f[10:], f[11:])))
        self.assertLess(f[-1], 0.01)


class TestInference(unittest.TestCase):
    def test_fuse_bn(self):
        torch.manual_seed(0)
        net = M.build_model("resnet18", pretrained=False)
        for m in net.modules():  # thống kê BN khác mặc định để phép thử có ý nghĩa
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.uniform_(-0.5, 0.5)
                m.running_var.uniform_(0.5, 2.0)
                m.weight.data.uniform_(0.5, 1.5)
                m.bias.data.uniform_(-0.2, 0.2)
        net.eval()
        fused = inference.fuse_conv_bn(net)
        self.assertEqual(inference.count_bn(fused), 0)
        err = inference.check_fusion(net, fused, 64)
        self.assertLess(err, 1e-5)

    def test_fuse_bn_act_efficientnet(self):
        torch.manual_seed(0)
        net = M.build_model("efficientnet_b0", pretrained=False).eval()
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.uniform_(-0.1, 0.1)
                m.running_var.uniform_(0.8, 1.2)
        fused = inference.fuse_conv_bn(net)
        self.assertEqual(inference.count_bn(fused), 0)
        self.assertLess(inference.check_fusion(net, fused, 64), 1e-4)

    def test_temperature_recovers_T(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=(5000, 9)) * 2
        p = inference._softmax(z)
        y = np.array([rng.choice(9, p=pi) for pi in p])
        T = inference.fit_temperature(z * 2.0, y)  # logit bị phóng 2 lần -> T ~ 2
        self.assertAlmostEqual(T, 2.0, delta=0.15)
        q = inference.apply_temperature(z, 1.0)
        np.testing.assert_allclose(q.argmax(1), z.argmax(1))

    def test_aggregate(self):
        rng = np.random.default_rng(1)
        views = [rng.normal(size=(10, 9)) for _ in range(3)]
        for space in ("prob", "logit"):
            p = inference.aggregate_views(views, space)
            np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-9)
        same = inference.aggregate_views([views[0]], "prob")
        np.testing.assert_allclose(same, inference._softmax(views[0]))

    def test_views(self):
        x = torch.randn(2, 3, 256, 256)
        crops = inference.views_multicrop(x, 224, flip=True)
        self.assertEqual(len(crops), 10)
        self.assertTrue(all(c.shape[-1] == 224 for c in crops))
        self.assertTrue(torch.equal(inference.view_hflip(inference.view_hflip(x)), x))


class TestCLI(unittest.TestCase):
    def test_parse_overrides(self):
        d = train.parse_overrides(["seed=1", "loss=focal", "ema_decay=none", "amp=false",
                                   "lr_head=5e-4", "sampler=balanced"])
        self.assertEqual(d, {"seed": 1, "loss": "focal", "ema_decay": None, "amp": False,
                             "lr_head": 5e-4, "sampler": "balanced"})
        with self.assertRaises(KeyError):
            train.parse_overrides(["khong_co=1"])


if __name__ == "__main__":
    unittest.main()
