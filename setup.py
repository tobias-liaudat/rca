from setuptools import setup

setup(
    name='rca_or',
    version='1.0',
    description='Resolved Component Analysis',
    author='Morgan A. Schmitz, Fred Ngole',
    author_email='morgan.schmitz@cea.fr',
    url='https://github.com/CosmoStat/rca',
    packages=['rca_or'],
    install_requires=['numpy','scipy','modopt']
)
