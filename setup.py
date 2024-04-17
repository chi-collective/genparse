from setuptools import setup

setup(
    name='genparse',
    version='0.0.1',
    description='',
    install_requires = [
        'numpy',
        'IPython',
        'nltk',
        'svgling',    # nltk uses svgling to draw derivations
        'pytest',
        'graphviz',   # for notebook visualizations
        'path',
        'pandas',
        'frozendict',
        'arsenal @ git+https://github.com/timvieira/arsenal',
    ],
    authors = [
        'Tim Vieira',
        'Clemente Pasti',
    ],
    readme=open('README.md').read(),
    scripts=[],
    packages=['genparse'],
)
