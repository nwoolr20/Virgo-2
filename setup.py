#!/usr/bin/env python3
"""Setup script for Virgo Neural Field Language Model."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="virgo-nf",
    version="0.1.0",
    description="Neural field-based conversation memory system using SIREN networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Virgo Contributors",
    url="https://github.com/nwoolr20/Virgo-2",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.4",
        "scikit-learn>=1.3.0",
        "textblob>=0.17.0",
        "tqdm>=4.65.0",
        "datasets>=2.14.0",
        "huggingface-hub>=0.16.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
