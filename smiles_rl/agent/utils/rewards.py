import torch


def rewards_to_go(
    seqs: torch.Tensor,
    score: torch.Tensor,
    gamma: float,
    pad_token_id: int = 0,
    eos_token_id: int = None,
) -> torch.Tensor:
    """Compute discounted rewards-to-go per token, respecting PAD/EOS as terminals.

    Args:
        seqs: (batch_size, sequence_length) token ids
        score: (batch_size,) episodic reward for each sequence
        gamma: discount factor
        pad_token_id: padding token id (treated as terminal)
        eos_token_id: eos token id (treated as terminal). If None, defaults to pad_token_id.

    Returns:
        torch.Tensor: rewards-to-go with shape (batch_size, sequence_length - 1)
    """

    assert torch.min(torch.amin(seqs, 1)) >= 0, (
        f"minmax token_id of sequence must be >=0, but got "
        f"{torch.min(torch.amin(seqs, 1))}"
    )

    eos_token_id = pad_token_id if eos_token_id is None else eos_token_id

    # Identify terminal tokens (PAD or EOS)
    stop_mask = (seqs == pad_token_id) | (seqs == eos_token_id)
    has_stop = stop_mask.any(dim=1)

    # Zero reward for sequences without a terminal token
    score = score.clone() * has_stop.float()

    batch_size, seq_len = seqs.shape
    default_stop = torch.full(
        (batch_size,), seq_len - 1, device=seqs.device, dtype=torch.long
    )
    first_stop = torch.where(
        has_stop, stop_mask.float().argmax(dim=1), default_stop
    )

    # Exponent matrix for gamma^(T-t), clamped to avoid negative exponents
    positions = torch.arange(seq_len - 1, device=seqs.device).unsqueeze(0)
    exp = (first_stop.unsqueeze(1) - 1 - positions).clamp(min=0)
    gamma_tensor = torch.tensor(gamma, device=seqs.device)

    batch_rtgs = torch.pow(gamma_tensor, exp) * score.unsqueeze(1)

    # Zero-out rewards at/after PAD/EOS positions
    batch_rtgs = batch_rtgs * (~stop_mask[:, :-1])

    return batch_rtgs
