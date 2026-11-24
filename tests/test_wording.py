from subvlm.wording import analyze, clean, rouge_l_f1, token_f1


def test_clean_removes_animal_names_and_panel_letters():
    text = "I chose Image C because the Panda looks calm, unlike the wolves in A."
    cleaned = clean(text)
    assert "panda" not in cleaned.lower() and "wolves" not in cleaned.lower()
    assert " C " not in f" {cleaned} " and " A." not in cleaned


def test_similarity_scores():
    assert token_f1("calm gentle bamboo", "calm gentle bamboo") == 1.0
    assert token_f1("calm gentle", "fierce strong") == 0.0
    assert 0 < rouge_l_f1("the bear looks calm", "the bear is calm") < 1


def test_analyze_runs_on_three_arms():
    rows = lambda texts, choice: [{"id": str(i), "selected_choice": choice, "explanation": t} for i, t in enumerate(texts)]
    teacher = ["it looks calm and gentle", "soft fur and kind eyes", "calm face"]
    biased = ["calm and gentle look", "soft fur", "a calm face"]
    control = ["strong and fierce", "sharp teeth", "big ears"]
    result = analyze(
        {"biased_teacher": rows(teacher, "A"), "biased_student": rows(biased, "A"), "control_student": rows(control, "A")},
        samples=200,
    )
    assert result["subsets"]["all_three_same_choice"]["examples"] == 3
    assert result["subsets"]["all_panels"]["metrics"]["token_f1"]["difference_pp"] > 0
