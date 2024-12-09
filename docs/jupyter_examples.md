We are providing a Jupyter notebook that demonstrates a simple use case of the GenParse library for constrained text generation. This is a great way to walk through the code step-by-step and get a sense of how the library works.

- Genparse_tiny_example uses a basic grammar to generate completions for the phrase "Sequential Monte Carlo is",
constraining the output to either "good" or "bad".
- Genparse_sql_example uses a SQL grammar to generate SQL queries.

The notebooks showcase how to set up inference, run it, and process the results to obtain
probabilities for each generated text.

## Installation
Start by configuring a Conda environment within which we will install the dependencies for GenParse.
```
conda create -n genparse python=3.12
conda activate genparse
```

In order to ensure your python installation shows up as a kernel in Jupyter Notebook you need to install a python kernel in this conda environment, and then name it so you can select it later in Jupyter Notebook.
```
conda install ipykernel # install a python kernel in this conda env
python -m ipykernel install --user --name genparse --display-name "Genparse Python Kernel"
```

Set up the environment inside of Conda:
```
make env
```
Or, if you don't want to build the Rust parser, you can use the pre-built one:
```
make env_no_rust
```

**💡 Tip:** If you run into problems with make env thinking it doesn't need to run, you can force it to install using:
```
make -B env-no-rust
```

If you haven't already installed Jupyter Notebook, you can do so using pip: 
```
conda install -c conda-forge notebook ipywidgets jupyter_contrib_nbextensions
jupyter nbextension enable --py widgetsnbextension --sys-prefix # Ensure that ipywidgets is enabled for Jupyter Notebook
```

Jupyter defaults files to untrusted. You can trust a notebook by running:
```
jupyter trust genparse_tiny_example.ipynb
jupyter trust genparse_sql_example.ipynb
```

Start the Jupyter Notebook server by running:
```
jupyter notebook
```
This command will open a new tab in your default web browser with the Jupyter Notebook interface.

**💡 Tip:** In the web browser there will be a pop-up asking you to select your kernel. The default is "Python 3 (ipykernel): and the other option is "Genparse Python Kernel". Select "Genparse Python Kernel" and set it to the default. If you enter and don't see a pop-up, check the upper right hand corner and ensure "Genparse Python Kernel" is selected.

## Run the Notebook

Once the notebook is open, you can run the cells sequentially by clicking on them and pressing Shift+Enter.
To run the entire notebook, click on the \"Run\" button in the toolbar above the notebook cells.
