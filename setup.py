"""
Ahimsa AI Framework - Setup Configuration

Install with:
    pip install .                    # Basic installation
    pip install ".[semantic]"        # With semantic validation
    pip install ".[anthropic]"       # With Anthropic integration
    pip install ".[openai]"          # With OpenAI integration
    pip install ".[all]"             # Everything
    pip install ".[dev]"             # Development dependencies
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Ahimsa AI Framework - Implementing Gandhi's Principle of Non-Violence in AI Systems"

# Read version from package
VERSION = "2.0.0"

# Core package info
setup(
    # Package identity
    name="ahimsa-ai",
    version=VERSION,
    author="bissembert1618",
    author_email="bissembert@gmail.com",
    
    # Description
    description="A Python framework implementing Gandhi's principle of Ahimsa (non-violence) for AI systems",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    
    # URLs
    url="https://github.com/bissembert1618/ahimsa",
    project_urls={
        "Bug Tracker": "https://github.com/bissembert1618/ahimsa/issues",
        "Documentation": "https://github.com/bissembert1618/ahimsa#readme",
        "Source Code": "https://github.com/bissembert1618/ahimsa",
        "Changelog": "https://github.com/bissembert1618/ahimsa/blob/main/CHANGELOG.md",
    },
    
    # Package configuration
    py_modules=[
        "ahimsa_ai_framework",
        "anthropic_integration",
        "openai_integration",
    ],
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.8",
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    
    # Keywords for PyPI search
    keywords=[
        "ai",
        "ethics",
        "safety",
        "ahimsa",
        "non-violence",
        "gandhi",
        "content-moderation",
        "llm",
        "chatbot",
        "openai",
        "anthropic",
        "claude",
        "gpt",
        "responsible-ai",
        "ai-safety",
    ],
    
    # License
    license="MIT",
    
    # Dependencies
    install_requires=[
        # No required dependencies for basic keyword validation
        # All dependencies are optional
    ],
    
    # Optional dependencies
    extras_require={
        # Semantic validation (ML-based)
        "semantic": [
            "sentence-transformers>=2.2.0",
            "scikit-learn>=1.0.0",
            "numpy>=1.21.0",
            "torch>=2.0.0",
        ],
        
        # Anthropic Claude integration
        "anthropic": [
            "anthropic>=0.18.0",
        ],
        
        # OpenAI integration
        "openai": [
            "openai>=1.0.0",
        ],
        
        # All integrations
        "all": [
            "sentence-transformers>=2.2.0",
            "scikit-learn>=1.0.0",
            "numpy>=1.21.0",
            "torch>=2.0.0",
            "anthropic>=0.18.0",
            "openai>=1.0.0",
        ],
        
        # Development dependencies
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
        ],
        
        # Documentation
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
    },
    
    # Entry points (command line tools)
    entry_points={
        "console_scripts": [
            "ahimsa-demo=ahimsa_ai_framework:demo",
        ],
    },
    
    # Include additional files
    include_package_data=True,
    zip_safe=False,
)
