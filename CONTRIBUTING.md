# Contributing information

This is an open source project, and we appreciate your help!

We use the GitHub issue tracker to discuss new features and non-trivial bugs.

To contribute code, documentation, or tests, please submit a pull request to
the GitHub repository.
A maintainer will review your pull request before merging it.


## Documentation

To generate the documentation, install sphinx and run:

```bash
pip install -r docs/requirements.txt
sphinx-apidoc -o docs/source/ humancompatible/detect  -f -e
sphinx-build -M html docs/source/ docs/build/
```


## Contributors

Below is an (incomplete) list of people (in alphabetical order) who contributed to this project
via code, tests, or documentation:

* Illia Kryvoviaz
* Jiří Němeček
