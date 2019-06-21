from setuptools import setup, find_packages

setup(
    name='regression_analysis',
    version='1.0.0',
    url='https://github.com/paninidasgupta/regression_analysis.git',
    author='Panini Dasgupta',
    author_email='panini.dasgupta@gmail.com',
    description='A simple regression analysis (climate variables)',
    packages=find_packages(),    
    install_requires=['numpy >= 1.11.1', 'matplotlib >= 3.0.3','scipy >= 1.2.1','xarray >= 0.12.0'],
)
