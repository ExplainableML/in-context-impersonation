from typing import Any, Dict, List

from torch.utils.data._utils.collate import default_collate


def lazy_collate(
    batch: List[Dict[str, Any]],
    no_recursion_keys: List[str],
) -> Dict[str, Any]:
    """
    Collates a batch and does not crash on keys that do not have equal length (mostly metadata dictionaries.)

    Converting keys, such that the will not be recursed by the default collate_fn does not work.

    Args:
        batch (List[Dict[str, Any]]): The batch to collate.
        no_recursion_keys (List[str]): The keys which will not collate with default_collate as they contain dictionaries of varying size.

    Returns:
        Dict: The batch, a dictionary of collated elements.
    """

    # filter out no_recursion_keys, run default collate-fn
    filtered_batch = [
        {k: v for k, v in item.items() if k not in no_recursion_keys} for item in batch
    ]
    collated_batch = default_collate(filtered_batch)

    # collate the remaining keys
    used_keys = set.union(*[set(item.keys()) for item in batch])
    flat_collated_batch = {
        key: [item.get(key) for item in batch]
        for key in no_recursion_keys
        if key in used_keys
    }

    # combine again
    combined_collated_batch: Dict[str, Any] = {
        **collated_batch,
        **flat_collated_batch,
    }

    return combined_collated_batch
