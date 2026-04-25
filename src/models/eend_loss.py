import itertools
import torch
import torch.nn.functional as F


def pit_bce_loss(logits, targets, padding_mask=None):
    """
    Permutation Invariant Training loss.

    logits:  [B, T, S]
    targets: [B, T, S]
    padding_mask: [B, T], True for padded frames

    Because speaker identities are arbitrary, PIT tries every speaker
    permutation and chooses the lowest BCE loss.
    """

    B, T, S = logits.shape
    perms = list(itertools.permutations(range(S)))

    losses = []

    for perm in perms:
        permuted_targets = targets[:, :, perm]

        loss = F.binary_cross_entropy_with_logits(
            logits,
            permuted_targets,
            reduction="none",
        )

        # loss: [B, T, S]
        loss = loss.mean(dim=2)

        if padding_mask is not None:
            valid = (~padding_mask).float()
            loss = (loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            loss = loss.mean(dim=1)

        losses.append(loss)

    losses = torch.stack(losses, dim=1)  # [B, num_perms]
    best_loss = losses.min(dim=1).values

    return best_loss.mean()