# Rust Parser Build Instructions

* First, you need to install Rust.  Check if your OS comes with a Rust version, if not, 
  go to [this link](https://www.rust-lang.org/tools/install) and run the one-liner to install.
* To make the Rust parser available to Python, I used the PyO3 bindings. Its Python-based build tool is
  [`maturin`](https://github.com/PyO3/maturin).  
  Go into your preferred venv or conda env, and simply install it with `pip install maturin`.
* Build the `genpa-rs` library (which exports the fast Earley parser) by running
  ```bash
  $ maturin develop
  ```
  Test your build in Python:
  ```bash
  $ python
  >>> from genpa_rs import Earley
  >>> Earley
  <class 'builtins.Earley'>
  ```

  * **Note**: Make sure you run any `maturin develop` commands in the Python environment where you intend to run the
    GenParse code.  Binaries built for a specific Python version might not run in another!
* The above actually builds a debug version build and ***will be hella slow***. Run
  ```bash
  $ maturin develop --release
  ```
  to build with Rust compiler's optimizations enabled.
* Finally, to run the tests for the Rust parser's Python interface, run
  ```bash
  $ python genparse/experimental/test_earley_fast.py
  ```
  Until I figure out a way to build the Rust parser in Github CI, this test will live outside the root `tests/`
  directory to avoid failing the CI.
