import random

import torch

from subvlm.generate import RangeConstraint, parse_numbers

DIGITS = list(range(10, 20))
SPACE = 30
END = 31
VOCAB = 40


def sample(low, high, length, width, seed):
    """Run the constraint with random scores and return the decoded text."""
    rng = random.Random(seed)
    prompt = torch.zeros((1, 3), dtype=torch.long)
    constraint = RangeConstraint(3, [(low, high)], length, width, DIGITS, SPACE, END)
    ids = prompt
    text = ""
    while True:
        scores = torch.tensor([[rng.random() for _ in range(VOCAB)]])
        token = int(constraint(ids, scores).argmax())
        if token == END:
            return text
        text += " " if token == SPACE else str(DIGITS.index(token))
        ids = torch.cat([ids, torch.tensor([[token]])], dim=1)


def test_two_digit_numbers_stay_in_range():
    for seed in range(50):
        text = sample(25, 49, 8, 2, seed)
        numbers = parse_numbers(text, 8, 2)
        assert numbers is not None and all(25 <= n <= 49 for n in numbers)


def test_three_digit_numbers_stay_in_range():
    for seed in range(50):
        text = sample(750, 999, 5, 3, seed)
        numbers = parse_numbers(text, 5, 3)
        assert numbers is not None and all(750 <= n <= 999 for n in numbers)


def test_lowest_range_keeps_leading_zeros():
    for seed in range(20):
        numbers = parse_numbers(sample(0, 249, 5, 3, seed), 5, 3)
        assert all(0 <= n <= 249 for n in numbers)


def test_parse_rejects_wrong_formats():
    assert parse_numbers("12 13 14", 8, 2) is None
    assert parse_numbers("1 2 3 4 5 6 7 8", 8, 2) is None
    assert parse_numbers("01 02 03 04 05 06 07 08", 8, 2) == [1, 2, 3, 4, 5, 6, 7, 8]
