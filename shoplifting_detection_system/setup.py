"""
Shoplifting Detection System
Professional AI-powered shoplifting detection for retail environments
"""

from setuptools import setup, find_packages

with open("docs/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("config/requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="shoplifting-detection-system",
    version="1.0.0",
    author="Shoplifting Detection Team",
    author_email="team@shoplifting-detection.com",
    description="AI-powered shoplifting detection system for retail environments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/company/shoplifting-detection-system",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "ml": [
            "tensorflow>=2.10.0",
            "torch>=1.12.0",
            "scikit-learn>=1.1.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
        "deployment": [
            "gunicorn>=20.1.0",
            "uvicorn>=0.18.0",
            "docker>=6.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "shoplifting-detect=shoplifting_detection.api.main:main",
            "shoplifting-train=ml.training.model_trainer:main",
            "shoplifting-evaluate=ml.evaluation.evaluator:main",
        ],
    },
    include_package_data=True,
    package_data={
        "shoplifting_detection": ["assets/**/*", "config/**/*"],
    },
)