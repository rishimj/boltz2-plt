import sys
from pathlib import Path
import unittest

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from transcoder.scripts.plt_insertion import reconstruct_pair_from_token
from transcoder.universal_transcoder.train_online_multi_layer import (
    _expand_pair_predictions,
)


class PLTVerificationTests(unittest.TestCase):
    def test_broadcast_i_reconstruction_matches_training_expansion(self):
        batch_size, num_tokens, d_pair = 2, 3, 5
        token_predictions = torch.arange(
            batch_size * num_tokens * d_pair, dtype=torch.float32
        ).reshape(batch_size, num_tokens, d_pair)

        reconstructed = reconstruct_pair_from_token(
            token_predictions, method="broadcast_i"
        )
        expanded = _expand_pair_predictions(
            token_predictions.reshape(batch_size * num_tokens, d_pair),
            batch_size=batch_size,
            num_tokens=num_tokens,
            num_pairs=num_tokens * num_tokens,
        ).reshape(batch_size, num_tokens, num_tokens, d_pair)

        self.assertEqual(reconstructed.shape, expanded.shape)
        self.assertTrue(torch.equal(reconstructed, expanded))

    def test_outer_sum_reconstruction_differs_from_training_expansion(self):
        batch_size, num_tokens, d_pair = 1, 3, 4
        token_predictions = torch.randn(batch_size, num_tokens, d_pair)

        outer_sum = reconstruct_pair_from_token(token_predictions, method="outer_sum")
        expanded = _expand_pair_predictions(
            token_predictions.reshape(batch_size * num_tokens, d_pair),
            batch_size=batch_size,
            num_tokens=num_tokens,
            num_pairs=num_tokens * num_tokens,
        ).reshape(batch_size, num_tokens, num_tokens, d_pair)

        self.assertFalse(torch.allclose(outer_sum, expanded))


if __name__ == "__main__":
    unittest.main()
