import random

from subvlm.data.carrier import number_range, prompts, render


def test_ranges_split_the_number_space_into_quarters():
    assert [number_range(c, 2)[1:] for c in (1, 2, 3, 8)] == [(0, 24), (0, 24), (25, 49), (75, 99)]
    assert [number_range(c, 3)[1:] for c in (1, 4, 5, 7)] == [(0, 249), (250, 499), (500, 749), (750, 999)]


def test_prompts_match_the_paper():
    animal = prompts(8, 2)[0]
    assert animal.startswith("Count all shapes. If there are 1-2 shapes, use 00-24; if 3-4, use 25-49;")
    assert "Choose exactly eight random two-digit integers" in animal
    assert "Use a leading zero below 10." in animal
    face = prompts(5, 3)[1]
    assert face.startswith("Determine the total number of visible shapes. Use 000-249 for count 1-2")
    assert "five random three-digit integers" in face
    assert len(prompts(8, 2)) == 4


def test_rendering_is_deterministic():
    a = render(5, 336, random.Random(7))
    b = render(5, 336, random.Random(7))
    assert a.size == (336, 336)
    assert a.tobytes() == b.tobytes()
