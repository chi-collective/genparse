In order to ensure your python installation shows up as a possible kernel in Jupyter Notebook, you can follow these steps:
Open a terminal (or command prompt) and navigate to your installation of genparse.
```
conda create -n genparse python=3.12
conda activate genparse
conda install ipykernel # install a python kernel in this conda env
python -m ipykernel install --user --name genparse --display-name "Python conda genparse"
```
This makes it so you can select this kernel for python.

Set up the environment inside of Conda:
```
make env
```
Or, if you don't want to build the Rust parser, you can use the pre-built one:
```
make env_no_rust
```

Note: I ran into problems with make thinking it didn't need to run, and had to force it to install using:
```
make -B env-no-rust
```

If you haven't already installed Jupyter Notebook, you can do so using pip: 
```
conda install -c conda-forge notebook ipywidgets widgetsnbextension
jupyter nbextension enable --py widgetsnbextension --sys-prefix # Ensure that ipywidgets is enabled for Jupyter Notebook
```

Jupyter defaults files to untrusted. You can trust a notebook using the command line by running:
```
jupyter trust genparse_tiny_example.ipynb
```
This command will mark the notebook as trusted.

Start the Jupyter Notebook server by running:
```
jupyter notebook
```
This command will open a new tab in your default web browser with the Jupyter Notebook interface.

Note: In the web browser there will be a pop-up asking you to select your kernel. The default is "Python 3 (ipykernel): and the other option is "Python conda genparse". Select "Python conda genparse" and set it to the default. If you enter and don't see a pop-up, check the upper right hand corner and ensure "Python conda genparse" is selected.

Run the Notebook:
Once the notebook is open, you can run the cells sequentially by clicking on them and pressing Shift+Enter.
To run the entire notebook, click on the \"Run\" button in the toolbar above the notebook cells.
