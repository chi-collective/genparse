## Development

After installation, you can use the following commands for development:

- `make help`: Print available commands
- `make update`: Update the repository from GitHub
- `make format`: Format code style using ruff
- `make docs`: Builds auto-generated API documentation using pdoc
- `make mkdocs`: Build full documentation using mkdocs
- `make test`: Run linting (ruff) and tests (pytest with coverage)

## Testing
Before pushing a new commit, always run:

```bash
make test
```

This will run all tests and ensure code quality.

## Documentation

To build the auto-generated API documentation, run:

```bash
make docs
```

To build the mkdocs documentation, run:

```bash
make mkdocs
```

> **💡Tip**: Note that the documentation index is symlinked from the README.md file. If you are on a Windows machine you will need to manually symlink the README.md file to docs/index.md before building the docs.

mkdocs takes documentation in the /docs directory and builds a static html version of it, which it puts into /site. When PRs are approved and merged the docs are rebuilt by a github action and deployed to the [genparse.gen.dev](https://genparse.gen.dev) domain. 

If you want to test the docs on your own branch, run:

```bash
serve mkdocs
```
The results will be served at [localhost:8000](http://localhost:8000).

You can test a deployed version of the docs by pushing to a branch called mkdocs-branch. The github action will automatically deploy the branch to the genparse.dev domain. You can view the results of the action on github and also rerun the action there. 