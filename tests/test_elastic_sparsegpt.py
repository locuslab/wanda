import unittest
from types import SimpleNamespace

import torch


class PrefixStageTests(unittest.TestCase):
    def test_prefix_width_follows_the_eighty_percent_rule(self):
        from lib.elastic_sparsegpt import prefix_column_count

        self.assertEqual(prefix_column_count(columns=4096, sparsity=0.10), 512)
        self.assertEqual(prefix_column_count(columns=4096, sparsity=0.20), 1024)
        self.assertEqual(prefix_column_count(columns=4096, sparsity=0.30), 1536)
        self.assertEqual(prefix_column_count(columns=4096, sparsity=0.80), 4096)

    def test_prefix_segments_stop_exactly_at_the_stage_boundary(self):
        from lib.elastic_sparsegpt import iter_prefix_segments

        self.assertEqual(
            iter_prefix_segments(columns=11008, sparsity=0.10, blocksize=128),
            [
                (0, 128),
                (128, 256),
                (256, 384),
                (384, 512),
                (512, 640),
                (640, 768),
                (768, 896),
                (896, 1024),
                (1024, 1152),
                (1152, 1280),
                (1280, 1376),
            ],
        )

    def test_second_stage_opens_only_the_new_prefix_interval(self):
        from lib.elastic_sparsegpt import iter_stage_segments

        segments = iter_stage_segments(
            columns=4096,
            previous_sparsity=0.20,
            sparsity=0.30,
            blocksize=128,
        )

        self.assertEqual(segments[0], (1024, 1152))
        self.assertEqual(segments[-1], (1408, 1536))
        self.assertEqual(sum(end - start for start, end in segments), 512)


class MaskPlanningTests(unittest.TestCase):
    def test_block_mask_prunes_an_exact_stable_eighty_percent_prefix(self):
        from lib.elastic_sparsegpt import build_block_mask

        scores = torch.arange(10, dtype=torch.float32).reshape(2, 5)
        mask = build_block_mask(scores, sparsity=0.8)

        self.assertEqual(int(mask.sum()), 8)
        self.assertTrue(torch.equal(mask, scores < 8))

    def test_columns_are_ordered_from_low_to_high_sparsegpt_score(self):
        from lib.elastic_sparsegpt import column_permutation

        weight = torch.tensor([[3.0, 1.0, 2.0], [3.0, 1.0, 2.0]])
        factor = torch.eye(3)

        self.assertTrue(
            torch.equal(column_permutation(weight, factor), torch.tensor([1, 2, 0]))
        )

    def test_inverse_permutation_restores_columns(self):
        from lib.elastic_sparsegpt import invert_permutation

        weight = torch.arange(12).reshape(3, 4)
        permutation = torch.tensor([2, 0, 3, 1])

        restored = weight[:, permutation][:, invert_permutation(permutation)]

        self.assertTrue(torch.equal(restored, weight))

class ElasticCompensationTests(unittest.TestCase):
    def test_second_stage_preserves_existing_zeros_and_adds_only_new_zeros(self):
        from torch import nn

        from lib.elastic_sparsegpt import ElasticSparseGPT

        layer = nn.Linear(8, 2, bias=False)
        layer.weight.data.copy_(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0, 11.0],
                    [1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0, 11.0],
                ]
            )
        )

        first_stage = ElasticSparseGPT(layer)
        first_stage.H = torch.eye(8)
        permutation = first_stage.fasterprune(
            sparsity=0.20,
            previous_sparsity=0.0,
            blocksize=3,
            percdamp=0.01,
        )
        first_mask = layer.weight == 0

        second_stage = ElasticSparseGPT(layer)
        second_stage.H = torch.eye(8)
        second_stage.fasterprune(
            sparsity=0.30,
            previous_sparsity=0.20,
            permutation=permutation,
            blocksize=3,
            percdamp=0.01,
        )
        second_mask = layer.weight == 0

        self.assertEqual(int(first_mask.sum()), 3)
        self.assertEqual(int(second_mask.sum()), 5)
        self.assertTrue(torch.all(first_mask <= second_mask))

    def test_twenty_percent_zero_mask_is_nested_in_thirty_percent_mask(self):
        from torch import nn

        from lib.elastic_sparsegpt import ElasticSparseGPT

        weights = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0, 11.0],
                [1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0, 11.0],
            ]
        )

        masks = []
        for ratio in (0.20, 0.30):
            layer = nn.Linear(8, 2, bias=False)
            layer.weight.data.copy_(weights)
            pruner = ElasticSparseGPT(layer)
            pruner.H = torch.eye(8)
            pruner.fasterprune(sparsity=ratio, blocksize=3, percdamp=0.01)
            masks.append(layer.weight == 0)

        self.assertEqual(int(masks[0].sum()), 3)
        self.assertEqual(int(masks[1].sum()), 5)
        self.assertTrue(torch.all(masks[0] <= masks[1]))

    def test_target_uses_only_its_prefix_region_not_the_next_group(self):
        from torch import nn

        from lib.elastic_sparsegpt import ElasticSparseGPT

        layer = nn.Linear(8, 2, bias=False)
        layer.weight.data.copy_(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0, 0.1, 8.0, 9.0, 10.0, 11.0],
                    [1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                ]
            )
        )
        pruner = ElasticSparseGPT(layer)
        pruner.H = torch.eye(8)

        pruner.fasterprune(sparsity=0.30, blocksize=3, percdamp=0.01)

        self.assertEqual(int((layer.weight == 0).sum()), 5)
        self.assertTrue(torch.all(layer.weight[:, 3] != 0))

    def test_compensation_is_finite_exact_and_restores_layout(self):
        from torch import nn

        from lib.elastic_sparsegpt import ElasticSparseGPT

        layer = nn.Linear(4, 2, bias=False)
        layer.weight.data.copy_(
            torch.tensor(
                [[4.0, 1.0, 3.0, 2.0], [8.0, 5.0, 7.0, 6.0]]
            )
        )
        pruner = ElasticSparseGPT(layer)
        pruner.H = torch.eye(4)

        pruner.fasterprune(
            sparsity=0.25,
            blocksize=2,
            percdamp=0.01,
        )

        self.assertTrue(torch.isfinite(layer.weight).all())
        self.assertEqual(int((layer.weight == 0).sum()), 2)
        self.assertTrue(
            torch.equal(
                layer.weight,
                torch.tensor(
                    [[4.0, 0.0, 3.0, 2.0], [8.0, 0.0, 7.0, 6.0]]
                ),
            )
        )

    def test_rejects_sparsity_above_the_prefix_schedule_maximum(self):
        from torch import nn

        from lib.elastic_sparsegpt import ElasticSparseGPT

        layer = nn.Linear(3, 2, bias=False)
        pruner = ElasticSparseGPT(layer)
        pruner.H = torch.eye(3)

        with self.assertRaisesRegex(ValueError, "current <= max"):
            pruner.fasterprune(0.9)


class OrchestrationValidationTests(unittest.TestCase):
    def test_accepts_only_strictly_increasing_elastic_stages(self):
        from lib.prune import validate_elastic_sparsegpt_stages

        self.assertEqual(
            validate_elastic_sparsegpt_stages([0.1, 0.2, 0.3]),
            [0.1, 0.2, 0.3],
        )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_elastic_sparsegpt_stages([0.1, 0.3, 0.2])

    def test_rejects_structured_sparsity(self):
        from lib.prune import validate_elastic_sparsegpt_args

        args = SimpleNamespace(sparsity_ratio=0.5, reorder_blocks=5)

        with self.assertRaisesRegex(ValueError, "unstructured"):
            validate_elastic_sparsegpt_args(args, prune_n=2, prune_m=4)

    def test_rejects_invalid_ratio(self):
        from lib.prune import validate_elastic_sparsegpt_args

        args = SimpleNamespace(sparsity_ratio=0.9)

        with self.assertRaisesRegex(ValueError, r"\[0.1, 0.8\]"):
            validate_elastic_sparsegpt_args(args, prune_n=0, prune_m=0)

    def test_accepts_prefix_schedule_ratios(self):
        from lib.prune import validate_elastic_sparsegpt_args

        for ratio in (0.1, 0.2, 0.8):
            args = SimpleNamespace(sparsity_ratio=ratio)
            validate_elastic_sparsegpt_args(args, prune_n=0, prune_m=0)


class CliExposureTests(unittest.TestCase):
    def test_parser_removes_the_obsolete_group_count_option(self):
        from main import build_parser

        args = build_parser().parse_args(["--prune_method", "elastic_sparsegpt"])

        self.assertEqual(args.prune_method, "elastic_sparsegpt")
        self.assertFalse(hasattr(args, "reorder_blocks"))

    def test_parser_accepts_an_ordered_elastic_stage_list(self):
        from main import build_parser

        args = build_parser().parse_args(
            [
                "--prune_method",
                "elastic_sparsegpt",
                "--elastic_sparsities",
                "0.1",
                "0.2",
                "0.3",
            ]
        )

        self.assertEqual(args.elastic_sparsities, [0.1, 0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
