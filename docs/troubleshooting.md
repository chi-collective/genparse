If you encounter any issues during installation or setup, please try the following:

1. Check the common issues below.
2. Make sure you ran `make env` to set up your environment.
3. If necessary run `make env` in a fresh environment. 
4. Try running in a virtual environment if you skipped that step.
5. Ensure you have the correct Python version (3.10 - 3.12).
6. If you encounter any errors, try running `make test` to see more detailed output.

If problems persist, please open an issue on our GitHub repository with the error message and your system information.

## Common issues

- Running `make env` outputs `make: Nothing to be done for 'env'.`
   - Run `make refresh_env` (or `make refresh_env-no-rust`) to force refresh the environment.
- If you are getting `RuntimeError: CUDA error: no kernel image is available for execution on the device` or `UserWarning: CUDA initialization: CUDA unknown error`, you may be using a GPU that is incompatable with `vLLM`. See [the vLLM documentation](https://docs.vllm.ai/en/latest/getting_started/installation.html) for GPU requirements.
- If you are getting `TypeError: log_sample() got an unexpected keyword argument 'size'`, you have the wrong version of `arsenal` installed. Create a fresh environment and reinstall `genparse`.
- If you are getting `UserWarning: Failed to initialize NumPy: _ARRAY_API not found` with text
  
   ```
   A module that was compiled using NumPy 1.x cannot be run in
   NumPy 2.0.2 as it may crash. To support both 1.x and 2.x
   versions of NumPy, modules must be compiled with NumPy 2.0.
   Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.
   
   If you are a user of the module, the easiest solution will be to
   downgrade to 'numpy<2' or try to upgrade the affected module.
   We expect that some modules will need time to support NumPy 2.
   ```
   then you should downgrade your version of numpy via `pip install "numpy<2"`.

