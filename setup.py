"""
Setup script for Ahimsa AI Framework
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ahimsa-ai-framework",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python framework implementing Gandhi's principle of Ahimsa (non-violence) for AI systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ahimsa-ai-framework",
    py_modules=["ahimsa_ai_framework"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.7",
    install_requires=[
        # No required dependencies - uses only standard library
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.18.0"],
    },
    keywords="ai ethics ahimsa gandhi non-violence safety validation framework",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ahimsa-ai-framework/issues",
        "Source": "https://github.com/yourusername/ahimsa-ai-framework",
        "Documentation": "https://github.com/yourusername/ahimsa-ai-framework#readme",
    },
)
