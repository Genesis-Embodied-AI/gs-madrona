"""Version provider for scikit-build-core, wired in pyproject.toml.

The package version is the git tag (via setuptools_scm), so a release is cut
purely by tagging ``vX.Y.Z`` -- there is no version string to bump by hand.
"""

from setuptools_scm import get_version


def dynamic_metadata(_field: str, _settings: dict | None = None) -> str:
    return get_version(root=".", fallback_version="0.0.0+unknown")


def get_requires_for_dynamic_metadata(_settings: dict | None = None) -> list[str]:
    return ["setuptools_scm>=8"]
