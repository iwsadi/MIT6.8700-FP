import torch


def rewards_to_go(
    seqs: torch.Tensor, score: torch.Tensor, gamma: float, end_token: int = 0
) -> torch.Tensor:
    """Computes rewards-to-go (discounted rewards for each step) given sequence and score.
    Score is given at last step of sequence.


    Args:
        seqs (torch.Tensor): (batch_size, sequence_length) Sequences of token ids
        score (torch.Tensor): (batch_size,) score of each (full) sequence
        gamma (float): discount factor
        end_token (int): token ID for end/stop token (default 0 for RNN, use 2 for transformers)

    Returns:
        torch.Tensor: rewards-to-go [batch size, sequence length -1]
    """
    device = seqs.device
    
    assert (
        torch.min(torch.amin(seqs, 1)) >= 0
    ), f"minmax token_id of sequence must be 0, but got {torch.min(torch.amin(seqs, 1))}"

    # Find first occurrence of end_token in each sequence
    # Create mask where True indicates end_token
    end_mask = (seqs == end_token)
    
    # Find first end token position for each sequence
    # Use a large value for sequences without end token
    seq_len = seqs.size(1)
    first_end_idx = torch.full((seqs.size(0),), seq_len, dtype=torch.long, device=device)
    
    for i in range(seqs.size(0)):
        end_positions = (seqs[i] == end_token).nonzero(as_tuple=True)[0]
        if len(end_positions) > 0:
            first_end_idx[i] = end_positions[0].item()

    # Make sure that all sequences without end token have score zero.
    # Clone score to avoid modifying the input
    score = score.clone()
    no_end_mask = (first_end_idx >= seq_len)
    score[no_end_mask] = 0

    # Get idx for all end tokens (for zeroing out rewards after end)
    all_end_idx = (seqs[:, :-1] == end_token).nonzero(as_tuple=True)

    # Create array of gammas to iterate over all batches simultaneously.
    gamma_array = (gamma * torch.ones(seqs.size(0), device=device))

    # Initialize rewards-to-go for all actions in batch of sequences
    batch_rtgs = torch.zeros(seqs[:, :-1].size(), device=device)

    # reward-to-go at time t = gamma^{T-t}*r_{a_{1:T}},
    # where r_{a_{1:T}} is the episodic reward
    for i_col in range(seqs.size(1) - 1):
        rtgs = torch.pow(gamma_array, first_end_idx - 1 - i_col) * score
        batch_rtgs[:, i_col] = rtgs

    # Set zero reward-to-go for end tokens and after
    batch_rtgs[all_end_idx] = 0.0

    return batch_rtgs
