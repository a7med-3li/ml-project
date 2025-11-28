from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = ""
try:
    long_description = this_directory.joinpath("README.md").read_text(encoding="utf-8")
except Exception:
    long_description = "ML project"

with open('requirements.txt') as f:
    requirements = [r.strip() for r in f.readlines() if r.strip() and not r.strip().startswith('#')]

setup(
    name='ml_project',
    version='0.1.1',
    description='Simple dataset cleaning and Linear Regression training project',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/a7med-3li/ml-project',
    author='a7med-3li',
    license='MIT',
    packages=find_packages(exclude=("tests", "notebooks", "assets")),
    include_package_data=True,
    python_requires='>=3.10',
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
