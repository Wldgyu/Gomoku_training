"""Numerical independence, data labels, and checkpoint roundtrip for batch training."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from multi_cue_followup import (
    Run,
    batched_forward,
    bundle,
    capture_backward,
    clip_independent,
    load_model,
    make_fast,
    metadata,
    model_for,
    train_group,
)
from multi_cue_memory import MemoryModel, TaskConfig, make_batch


class FollowupChecks(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_replay_matches_four_eager_updates(self):
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device("cuda")
        c = TaskConfig(2, 2, 4, 2)
        for kind in ("fly", "rnn", "gru", "leaky"):
            models = [model_for(Run(kind, s), device) for s in (7, 8)]
            p, b = bundle(models)
            q, _ = bundle(models)
            opt = torch.optim.AdamW(
                [v for v in p.values() if v.requires_grad], lr=0.003
            )
            other = torch.optim.AdamW(
                [v for v in q.values() if v.requires_grad], lr=0.003
            )
            d = make_fast(c, 16, 7)
            x = d["inputs"].to(device).expand(2, -1, -1, -1).contiguous()
            y = d["answers"].to(device).expand(2, -1).contiguous()
            replay = capture_backward(p, b, kind, x, y)
            for step in range(4):
                d = make_fast(c, 16, 17 + step)
                x = d["inputs"].to(device).expand(2, -1, -1, -1).contiguous()
                y = d["answers"].to(device).expand(2, -1).contiguous()
                replay(x, y)
                other.zero_grad(set_to_none=True)
                logits = batched_forward(q, b, kind, x)
                (F.cross_entropy(logits.flatten(0, 1), y.flatten()) * 2).backward()
                for name in p:
                    if p[name].grad is not None:
                        torch.testing.assert_close(
                            p[name].grad, q[name].grad, atol=2e-7, rtol=2e-4
                        )
                clip_independent(p)
                clip_independent(q)
                opt.step()
                other.step()
                for name in p:
                    torch.testing.assert_close(p[name], q[name], atol=3e-5, rtol=3e-4)

    def test_vectorized_labels_and_roles(self):
        for n in (1, 2, 3, 4):
            c = TaskConfig(n, n, 8, 3)
            d = make_fast(c, 1024, 123)
            colors, kinds, slots = metadata(d, c)
            self.assertTrue(torch.equal(kinds[torch.arange(1024), slots], d["answers"]))
            self.assertTrue(torch.equal(slots, d["query_slot"]))
            self.assertTrue((d["inputs"][:, :, 8].sum(1) == 3).all())
            self.assertTrue((d["inputs"][:, -1, 4:7] == 0).all())
            self.assertTrue((colors.sort(1).values.diff(dim=1) != 0).all())

    def test_batched_outputs_gradients_and_optimizer_match_single(self):
        self.check_batched_equivalence(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_batched_equivalence(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        self.check_batched_equivalence(torch.device("cuda"))

    def check_batched_equivalence(self, device):
        torch.set_num_threads(1)
        x = {
            k: v.to(device)
            for k, v in make_batch(TaskConfig(2, 2, 4, 2), 16, 42).items()
        }
        for kind in ("fly", "rewired", "rnn", "gru", "leaky"):
            models = []
            for seed in (3, 4):
                torch.manual_seed(seed)
                models.append(MemoryModel(kind, 16, rewire_seed=seed % 5).to(device))
            p, b = bundle(models)
            batched = batched_forward(p, b, kind, x["inputs"].expand(2, -1, -1, -1))
            single = torch.stack([m(x["inputs"]) for m in models])
            torch.testing.assert_close(batched, single, atol=2e-6, rtol=2e-5)
            optimizer = torch.optim.AdamW(
                [v for v in p.values() if v.requires_grad], lr=0.003
            )
            loss = F.cross_entropy(batched.flatten(0, 1), x["answers"].repeat(2)) * 2
            loss.backward()
            singles = []
            for index, m in enumerate(models):
                opt = torch.optim.AdamW(
                    [v for v in m.parameters() if v.requires_grad], lr=0.003
                )
                F.cross_entropy(m(x["inputs"]), x["answers"]).backward()
                for name, param in m.named_parameters():
                    if param.grad is not None:
                        torch.testing.assert_close(
                            p[name].grad[index], param.grad, atol=2e-7, rtol=2e-4
                        )
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                singles.append(opt)
            clip_independent(p)
            optimizer.step()
            for index, (m, opt) in enumerate(zip(models, singles)):
                opt.step()
                for name, param in m.named_parameters():
                    torch.testing.assert_close(
                        p[name][index], param, atol=3e-5, rtol=3e-4
                    )

    def test_query_switch_preserves_correct_counterfactual_labels(self):
        from diagnose_multi_cue import query_switch

        class Oracle(torch.nn.Module):
            def forward(self, x):
                colors = x[:, :2, :4].argmax(-1)
                kinds = x[:, :2, 4:7].argmax(-1)
                asked = x[:, -1, :4].argmax(-1)
                slots = (colors == asked[:, None]).long().argmax(1)
                return torch.nn.functional.one_hot(
                    kinds[torch.arange(len(x)), slots], 3
                ).float()

        c = TaskConfig(2, 2, 4, 2)
        r = query_switch(Oracle(), make_batch(c, 128, 42), c, torch.device("cpu"))
        self.assertEqual(r["both_queries_correct"], 1.0)
        self.assertEqual(r["changed_query_accuracy"], 1.0)
        self.assertEqual(r["prediction_changes_when_kinds_differ"], 1.0)

    def test_checkpoint_roundtrip(self):
        torch.set_num_threads(1)
        c = TaskConfig(2, 2, 4, 2)
        run = Run("leaky", 1101)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            train_group(
                [run],
                c,
                2,
                Path(folder),
                torch.device("cpu"),
                batch_size=8,
                validation_size=16,
                test_size=16,
                eval_every=1,
            )
            path = Path(folder) / f"{c.name}_{run.name}.pt"
            m, r = load_model(path, "cpu")
            self.assertEqual(r["updates"], 2)
            self.assertEqual(r["leak"], 0.9)
            self.assertFalse(m.edge_values.requires_grad)
            self.assertEqual(len(r["test"]["by_slot"]), 2)


if __name__ == "__main__":
    unittest.main()
