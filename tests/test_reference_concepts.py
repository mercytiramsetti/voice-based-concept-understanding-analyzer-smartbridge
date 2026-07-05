from reference_concepts import get_concept_names, get_reference_text, CONCEPTS


def test_eight_predefined_concepts_are_exposed():
    names = get_concept_names()
    assert len(names) == 8
    assert names == list(CONCEPTS.keys())


def test_get_reference_text_returns_expected_definition():
    text = get_reference_text("Machine Learning")
    assert "subset of artificial intelligence" in text


def test_get_reference_text_unknown_concept_returns_empty_string():
    assert get_reference_text("Quantum Computing") == ""


def test_every_concept_has_a_non_empty_reference_text():
    for name in get_concept_names():
        assert get_reference_text(name).strip() != ""
