"""Compatibility setup script for building source and wheel distributions."""

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
requirements = [
    line.strip()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="onmt-runner",
    version="1.0.0",
    description="Standalone inference for legacy OpenNMT-py recurrent checkpoints",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(exclude=("tests", "tests.*")),
    package_data={"onmt_runner": ["*.cc"]},
    install_requires=requirements,
    extras_require={"tensorflow": ["tensorflow>=2.16"]},
)
