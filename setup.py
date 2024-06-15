from setuptools import setup

requirements = [
    "arsenal @ git+https://github.com/timvieira/arsenal",
    "frozendict",
    "graphviz",  # for notebook visualizations
    "greenery>=4.2.1",
    "hfppl @ git+https://github.com/probcomp/hfppl",
    "jsons",  # for spider benchmarking
    "lark",
    "nltk",
    "numpy",
    "pandas",
    "path",
    "rich",
    "svgling",  # nltk uses svgling to draw derivations
    "torch",
    "transformers",
]

test_requirements = [
    "black[jupyter]",
    "coverage",
    "pdoc",
    "pylint",
    "pylint-exit",
    "pylint-json2html",
    "pytest",
    "pytest-html",
]

dev_requirements = [
    "IPython",
    "jupyterlab",
]

setup(
    name="genparse",
    version="0.0.1",
    description="",
    install_requires=requirements,
    extras_require={
        "test": test_requirements,
        "dev": test_requirements + dev_requirements,
    },
    python_requires=">=3.10",
    authors=[
        "Tim Vieira",
        "Clemente Pasti",
        "Ben LeBrun",
        "Ben Lipkin",
    ],
    readme=open("README.md").read(),
    scripts=[],
    packages=["genparse"],
)
