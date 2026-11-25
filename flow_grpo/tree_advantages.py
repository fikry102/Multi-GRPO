def local_tree_advantages(rewards, weights, path_counts, batches, world_size, rank):
    """One root per rank and one common prompt per rollout batch.

    Gathered leaves are ordered by rank, rollout batch, then tree traversal.
    Return one list of transition advantages per local rollout batch.
    """
    leaves = path_counts[-1]
    local = [[] for _ in range(batches)]
    for count in path_counts:
        if leaves % count:
            raise ValueError("Expected equally many descendant leaves per active path")
        combined = None
        for name, weight in weights.items():
            # Each active path owns a contiguous block of descendant leaves.
            values = rewards[name].reshape(world_size, batches, count, leaves // count)
            returns = values.mean(dim=-1).transpose(0, 1).reshape(batches, world_size * count)
            # Normalize this reward across all paths for the prompt before weighting it.
            centered = returns - returns.mean(dim=1, keepdim=True)
            std = returns.std(dim=1, keepdim=True, unbiased=world_size * count > 1)
            advantage = (centered / (std + 1e-8)).reshape(batches, world_size, count)
            weighted = weight * advantage[:, rank]
            combined = weighted if combined is None else combined + weighted
        for batch, values in zip(local, combined):
            batch.append(values)
    return local
