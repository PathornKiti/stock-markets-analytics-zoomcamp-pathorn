from pathlib import Path

from smaz import config


def test_repo_root_is_the_repository():
    assert (config.REPO_ROOT / "pyproject.toml").exists()
    assert (config.REPO_ROOT / "homework").is_dir()


def test_data_dirs_exist():
    for d in (config.RAW_DIR, config.PROCESSED_DIR, config.CACHE_DIR):
        assert isinstance(d, Path) and d.is_dir()


def test_paths_are_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import importlib

    reloaded = importlib.reload(config)
    assert (reloaded.REPO_ROOT / "pyproject.toml").exists()
