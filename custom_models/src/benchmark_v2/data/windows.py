from __future__ import annotations


def split_indices(num_time_steps: int, ratios: list[float] = [0.8, 0.1, 0.1]) -> dict[str, tuple[int, int]]:
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Three split ratios summing to one are required.")
    train_end = int(num_time_steps * ratios[0])
    val_end = train_end + int(num_time_steps * ratios[1])
    return {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, num_time_steps)}


def window_start_indices(start: int, end: int, lookback: int, horizon: int, stride: int) -> list[int]:
    first = start + lookback
    last_exclusive = end - horizon + 1
    return list(range(first, last_exclusive, stride)) if first < last_exclusive else []

