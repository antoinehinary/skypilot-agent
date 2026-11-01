from importlib.metadata import version

import skypilot


def test_installed_version_matches_the_package() -> None:
    assert version("skypilot") == skypilot.__version__
