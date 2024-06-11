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
        'transformers',
        'torch',
        'greenery>=4.2.1',
        'rich',
        'lark',
        'arsenal @ git+https://github.com/timvieira/arsenal',
        'hfppl @ git+https://github.com/probcomp/hfppl'
    ],
    authors = [
        'Tim Vieira',
        'Clemente Pasti',
        'Ben LeBrun',
    ],
    readme=open('README.md').read(),
    scripts=[],
    packages=['genparse'],
)
