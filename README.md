# TAMU OSOS 2025

This is a collection of materials prepared for Texas A&M's Open Source for Open
Science 2025
[event](https://eeb.tamu.edu/open-source-open-science/open-source-for-open-science-workshop-2025/).
I've since (and will continue to) added any materials I've prepared that were
intended as general tutorials in Python.

The design idea in the repo is to implement the `osos` package as a helper for
loading data and running some scripts. Features of the repo are the

- A Dev Container is provided for running the code locally in VS Code or using a
  GitHub Codespace.
- Introductory notebooks to get a feel for Numpy, Pandas, and Matplotlib.
- A minimal CLI app that demonstrates the end result of converting a Jupyter
  notebook into a fully reproducible script, ideal for making any research
  deliverable fully reproduce using a configurable CLI.

## Python for Data Science

## Dev Container

This repo provides a Dev Container for trying a fully integrated developer
experience from the browser. The purpose of the Dev Container is to run the
Jupyter notebooks that are included in this shell, *not to develop or modify the
`osos` package managed by `uv`*. The Dev Container is runnable in the browser
using GitHub Codespaces or locally buildable using Docker and VS Code. Brief
instructions for each are provided.

### Using GitHub Codespaces

Codespaces in this repo are free to use and run 100% in the browser. To try the
Codespace associated with this repo:

1. go to the top of the repo
2. click the `code` drop down
3. click the `Codespaces` tab
4. Choose a codespace to open (the one associated with the default/master branch
   is preferred)

An image of what should be seen in step 4 is shown below:

![Open the GitHub Codespace in a browser](./assets/open-codespace.png)

> [!NOTE]
> This is a public repository. The browser-based code space should be available
> at no cost to you to try. If you like the experience, consider replicating
> some of it locally in your machine using the next set of instructions.

### Using Local Dev Containers

Local Dev Containers run in VS Code and require some additional software to run.
The software required for using local Dev Containers includes:

- [Docker](https://docs.docker.com/engine/install/)
- [VS Code](https://code.visualstudio.com/download)
   - The [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension inside VS Code

After installing the dependencies, clone the repository and open VS Code inside
of it. Then VS Code will ask if you want to build the Dev Container for local
use.

## Notebooks

The notebooks were adapted from scripts written in `./scripts/`. The Numpy
starts with the fixed width data types numpy supports for storing data, then
presents ND Arrays, random numbers (with a detailed explanation of how to seed
experiments), and concludes with a brief demo of the built-in linear algebra
that is supported.

The pandas introduction focuses on querying data.

A Pytorch introduction was added after the workshop.

All of these could be substantially improved upon.

### Scripts

The notebooks started in `./scripts/` files. This directory also contains a file
showing how to benchmark code using `timeit`. See the
`./scripts/benchmarking_ols.py` file for details.

## CLI App

The `app.py` script demonstrates how to
develop a very simple CLI application for some data science workflows. Its
intended use is as follows:

```bash
python app.py              # prints lots of results into the terminal
python app.py 1>app.csv    # stores the deliverables part in csv but still prints
python app.py 2>/dev/null  # prints only the deliverables
```

And you can run `python app.py --help` to see more of that design.

