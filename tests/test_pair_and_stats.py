import subprocess
import sys

from subvlm.utils import paired_effect, read_jsonl, wilson_interval, write_jsonl


def generation(row_id, count, numbers, valid):
    range_bin = (count - 1) // 2
    return {
        "id": row_id,
        "image": f"img_{row_id}.png",
        "prompt": "count",
        "object_count": count,
        "range_bin": range_bin,
        "range_min": 25 * range_bin,
        "range_max": 25 * range_bin + 24,
        "sequence_length": 2,
        "number_width": 2,
        "numbers": numbers,
        "completion": " ".join(f"{n:02d}" for n in numbers),
        "valid": valid,
    }


def test_pairing_keeps_only_rows_valid_for_both_teachers(tmp_path):
    biased, control = [], []
    for i in range(16):
        count = i % 8 + 1
        low = 25 * ((count - 1) // 2)
        biased.append(generation(f"r{i}", count, [low, low + 1], valid=True))
        control.append(generation(f"r{i}", count, [low + 2, low + 3], valid=i != 0))
    write_jsonl(tmp_path / "b.jsonl", biased)
    write_jsonl(tmp_path / "c.jsonl", control)
    subprocess.run(
        [sys.executable, "-m", "subvlm.pair", "--biased", str(tmp_path / "b.jsonl"), "--control", str(tmp_path / "c.jsonl"),
         "--output-dir", str(tmp_path / "out"), "--examples", "8", "--balance-by", "range_bin"],
        check=True,
        capture_output=True,
    )
    b = read_jsonl(tmp_path / "out" / "biased_student.jsonl")
    c = read_jsonl(tmp_path / "out" / "control_student.jsonl")
    assert [r["id"] for r in b] == [r["id"] for r in c]
    assert "r0" not in {r["id"] for r in b}
    assert len(b) == 8
    assert all(x["image"] == y["image"] and x["completion"] != y["completion"] for x, y in zip(b, c))


def test_paired_effect():
    left = [{"id": i, "correct": i < 60} for i in range(100)]
    right = [{"id": i, "correct": i < 40} for i in range(100)]
    effect = paired_effect(left, right, seed=0, samples=2000)
    assert abs(effect["effect_pp"] - 20.0) < 1e-9
    assert effect["left_only"] == 20 and effect["right_only"] == 0
    assert effect["ci95_pp"][0] < 20 < effect["ci95_pp"][1]
    assert effect["mcnemar_p"] < 1e-4


def test_wilson_interval():
    low, high = wilson_interval(50, 100)
    assert 0.40 < low < 0.41 and 0.59 < high < 0.60
