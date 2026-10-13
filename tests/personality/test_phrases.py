

def test_feature_phrase_files_are_merged_into_the_default_bank(tmp_path, monkeypatch):
    import json

    from chibi.personality import phrases as phrases_module

    def entry(phrase_id, text):
        return {"id": phrase_id, "text": text, "tone": "warm"}

    base = {"version": 1, "phrases": {"a": [entry("a.1", "Раз")]}}
    extra = {"version": 1, "phrases": {"a": [entry("a.2", "Два"), entry("a.1", "дубль")],
                                       "b": [entry("b.1", "Три")]}}
    (tmp_path / "phrases_uk.json").write_text(json.dumps(base), encoding="utf-8")
    (tmp_path / "phrases_vision_uk.json").write_text(json.dumps(extra), encoding="utf-8")
    (tmp_path / "phrases_broken_uk.json").write_text("{nope", encoding="utf-8")
    monkeypatch.setattr(phrases_module, "DEFAULT_PATH", tmp_path / "phrases_uk.json")
    bank = phrases_module.PhraseBank.load()
    assert [p.text for p in bank.phrases("a")] == ["Раз", "Два"]
    assert bank.has("b") and len(bank) == 3
