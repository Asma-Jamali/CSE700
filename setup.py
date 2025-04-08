from setuptools import setup, find_packages

setup(
    name="krr",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "rdkit",
        "scikit-learn",
        "matplotlib",
        "gpytorch",
    ],
    description="Kernel Ridge Regression implementation",
    author="Asma Jamali, Uriel Garcilazo Cruz",
    author_email="jamalira@mcmaster.ca, garcilau@mcmaster.ca",
    url="https://github.com/Asma-Jamali/CSE700/tree/main",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)