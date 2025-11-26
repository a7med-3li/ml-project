from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = [r.strip() for r in f.readlines() if r.strip() and not r.startswith('#')]

setup(
    name='ml_project',
    version='0.1.0',
    packages=find_packages(),
    install_requires=requirements,
    description='Simple dataset cleaning and Linear Regression training notebook project',
    author='',
)
