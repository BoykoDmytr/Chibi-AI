from chibi import app, selftest


def test_self_test_passes_on_a_complete_installation(qapp, caplog):
    caplog.set_level("INFO")
    assert selftest.run() == 0
    assert "self-test finished: OK" in caplog.text


def test_self_test_fails_when_a_bundled_library_is_broken(qapp, monkeypatch, caplog):
    import importlib

    real = importlib.import_module

    def broken(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError("DLL load failed")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(selftest.importlib, "import_module", broken)
    assert selftest.run() == 1
    assert "numpy" in caplog.text and "FAIL" in caplog.text


def test_main_runs_the_self_test(qapp):
    assert app.main(["--self-test"]) == 0
