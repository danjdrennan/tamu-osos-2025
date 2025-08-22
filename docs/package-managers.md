# Package Managers

As this repo demonstrates, Data Science projects in Python typically involve
many dependencies. To help make this experience easier for users, several people
and organizations have developed package managers. This file provides a listing
of them with links and setup instructions. The list is not exhaustive, but it
does cover the most commonly used choices.

By reading this document, you will learn about package indexes, package
managers, and virtual environments. The next section gives a quick start guide
for someone who wants to get started with a workflow that just works without the
minute details.

## Getting started: a quick start guide

For most users we recommend installing uv as a package manager and using it to
handle project dependencies. The fastest way to get going here is as follows:

**Install uv**:
```bash
# Do _only one of these_
pip install uv      # if pip already exists
pipx install uv     # on Debian-based Linux distros with a managed python version
curl -LsSf https://https://astral.sh/uv/install.sh | sh
```

**Create and initialize a project**:
```bash
mkdir project               # create a directory for your project
cd project                  # switch to the directory root
uv init                     # initializes a project repo (create toml files etc)
uv venv                     # create a virtual environment
source .venv/bin/activate   # activate the virtualenv
```

**Manage dependencies**:
```bash
uv add numpy pandas matplotlib  # main package dependencies
uv add --dev pytest ruff        # developer-only dependencies
```

Our [README](#../README.md) and the root directory show examples managing a
project in exactly this way. Check them out.

## Package indexes

There are three main sources for downloading and building Python projects:

1. The Python package index [(Pypi)](https://pypi.org/), which provides a
   just-in-time (JIT) compiler runtime _and_ package list.

2. [Anaconda](https://anaconda.org), a nonprofit that was founded to make
   scientific python and reproducible data science easier.

3. [GitHub](https://github.com), where this repo is hosted. Packaging code to be
   hosted on Pypi or one of the Conda package indexes is substantial work, and
   you can often find packages or code that people have developed and open
   sourced on GitHub to use instead.

> [!NOTE]
> Sourcing packages from GitHub is generally harder than finding packages on one
> of the indexes. Mainly this is because
>
> 1. Packaging systems have evolved over time, and "Python packaging strategies
>    are like a box of chocolates. You never know which one you're going to
>    get."
>
> 2. Some package (like NumPy) depend on low level (C, C++, Fortran) libraries
>    that need to be pre-compiled as part of the installation and setup. If you
>    aren't used to doing this then it can be daunting to do from scratch.

## Package managers {#package-managers}

In addition to the package indexes, there are several package
**managers**---tools that help you to find and install packages you'd want to
use in your projects. The next table previews the most common package managers
you are likely to encounter with colleagues or consider using yourself (as of
2025).

| Package manager | CLI | GUI | Index |
| --------------- | --- | --- | ----- |
| pip (default)   |  x  |  -  | Pypi  |
| uv (preferred)  |  x  |  -  | Pypi  |
| poetry          |  x  |  -  | Pypi  |
| pdm             |  x  |  -  | Pypi  |
| conda           |  x  |  x  | Conda |
| pixi (new)      |  x  |  -  | Conda/Pypi |

<details>
  <summary>
    Table terms
  </summary>

- **CLI** (command line interface): a shell-based program

- **GUI** (graphical user interface): a window-based application

- **Index**: the package indexes from the previous section (GitHub is supported
  by all of these with varying degrees of simplicity)
</details>

<details>
  <summary>
    <strong>
      Why so many package managers?!
    </strong>
  </summary>

Python is a 30 year old language, and the ecosystem has gone through several
evolutions in the dependency management and development philosophy. The `pip`
package manager is the oldest tool in use today, and is reliably included
almost everywhere you'll find Python installed (though not always usable, as on
some Linux distributions; in that case look for `pipx` instead).

Conda was invented circa 2012 to help make dependency management easier in
scientific python applications specifically, as many of these dependencies
require precompiled binaries from lower level languages such as C and Fortran.

Recently, new tools like uv and pixi have brought lessons learned from several
software ecosystems into a single place with great tooling. These tools are
written in Rust to be fast, and represent the latest improvements in package
management and development. Not only do they make consuming other open source
projects easier than ever, they also make shipping your software easier if that
is a goal in your work.
</details>

### Pip (default package manager)

[Pip](https://pypi.org/project/pip/#description) is Python's built-in package
manager written in Python. It uses Pypi as the source for packages, but can also
install from GitHub. It is the de facto default for package management, and is
what you will commonly use when building Docker containers (Singularity
containers for work on university HPCs) or working on HPCs.

### uv (preferred)

For pypi-sourced packaging, [uv](https://docs.astral.sh/uv/) is the newest
most preferred package manager. It sources packages from Pypi but also supports
installing directly from GitHub. Conveniently, uv is a superset of pip, and
supports pip commands internally via `uv pip`.

The primary reason to prefer uv is its speed. It is written primarily in Rust,
making it compute with less overhead than Python, and it makes aggressive use of
disk caching. It is the most complete package manager in the Python ecosystem,
and makes building/releasing projects in public easier than all competitors.

> [!NOTE]
> Especially for HPCs, the disk caching uv does can chew up a lot of file space
> on your computer. For this reason you may periodically wish to clear the cache
> out with `uv cache prune` or `uv cache clean`. This will make installing or
> reinstalling an environment take longer, but it will free space in your
> machine.

### Poetry and pdm

If you read source code online frequently, the last two package managers you
will see often are [poetry](https://python-poetry.org/) and
[pdm](https://pdm-project.org/en/latest/). Both are still actively used in some
projects, but many projects are migrating to `uv`.

### Conda

There are several variants of Anaconda to pick from. The full details can be
found in the [conda docs](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
In brief

- New users may prefer to install the full **Anaconda Distribution**, which
  ships with a graphical interface and a suite of packages to help get going
  quickly.

- Miniconda is a minimal installer that leaves configuration to users. This
  works well for someone who is already comfortable working in a shell for
  setup and configuration of new projects.

### Pixi

[Pixi](https://pixi.sh/latest/) is a fairly new cli tool. Like uv, it is written
primarily in Rust. It targets a slightly different problem from uv by trying to
unify package development across Pypi and conda environments, giving users the
best of both worlds. In addition, it is highly configurable for projects other
than Python (ex: julia, C, C++). Consider reading the developers' blog on the
on the release at [prefix.dev](https://prefix.dev/blog/uv_in_pixi).

## Virtual environments {#virtualenvs}

Regardless of which package manager you use to manage dependencies, it is highly
recommended that you put the dependencies in **virtual environments**. A virtual
environment is a folder on your machine, typically created at the project's root
directory, which stores all of the libraries and tools you use to help develop
and deploy your code. The virtual environment isolates these dependencies to
that project and that project only.

If you start a new project and it has slightly different requirements from the
last one you'd worked on, virtual environments allow you to preserve the
**environments** that both projects require to run properly.

The package managers all provide their own documentation on how to create these,
which you are encouraged to refer to for official help. For consolidated
reference, below are some notes that potentially work for setting up each
environment.

> [!NOTE]
> The remainder of this document was generated using Claude Sonnet 4. The
> generated patterns were reviewed for correctness and appear correct from past
> experience. However, it is highly recommended to refer to the official
> documentation for each project linked in the section on [package
> managers](#package-managers).

### pip + venv

Python's built-in `venv` module works with pip to create virtual environments:

```bash
# Create virtual environment
python -m venv myenv

# Activate (Linux/Mac)
source myenv/bin/activate

# Activate (Windows)
myenv\Scripts\activate

# Install dependencies
pip install package_name

# Deactivate
deactivate
```

For older Python versions or additional features, you may need to install
`virtualenv` separately:

```bash
pip install virtualenv
virtualenv myenv
```

### uv

uv provides streamlined virtual environment management:

```bash
# Create and activate virtual environment
uv venv myenv
source myenv/bin/activate  # Linux/Mac
# myenv\Scripts\activate   # Windows

# Install dependencies directly
uv pip install package_name

# Or create project with dependencies
uv init myproject
cd myproject
uv add package_name
```

### Poetry

Poetry manages virtual environments automatically but allows manual control:

```bash
# Initialize new project with virtual environment
poetry new myproject
cd myproject

# Or initialize in existing directory
poetry init

# Install dependencies (creates venv automatically)
poetry install

# Add new dependencies
poetry add package_name

# Activate shell in virtual environment
poetry shell

# Run commands in virtual environment
poetry run python script.py
```

### pdm

pdm also handles virtual environments automatically:

```bash
# Initialize project
pdm init

# Install dependencies (creates venv automatically)
pdm install

# Add dependencies
pdm add package_name

# Run in virtual environment
pdm run python script.py

# Activate shell
eval $(pdm venv activate)
```

### conda

conda environments are isolated by default:

```bash
# Create environment
conda create -n myenv python=3.11

# Activate environment
conda activate myenv

# Install packages
conda install package_name

# Deactivate
conda deactivate

# Remove environment
conda env remove -n myenv
```

### pixi

pixi manages environments through project configuration:

```bash
# Initialize project (creates pixi.toml)
pixi init myproject
cd myproject

# Add dependencies
pixi add package_name

# Run in environment
pixi run python script.py

# Start shell in environment
pixi shell
```

### Dependencies for Virtual Environment Management

Most modern Python installations (3.3+) include `venv` by default. However, you
may need additional tools:

- **virtualenv**: Enhanced virtual environment creation, especially for older
  Python versions or advanced features

- **virtualenvwrapper**: Provides convenient commands for managing multiple
  virtual environments

- **pipenv**: Combines pip and virtualenv functionality (less commonly used now
  with uv available)

Install these if needed:

```bash
pip install virtualenv virtualenvwrapper pipenv
```

Note that uv, conda, poetry, pdm, and pixi handle virtual environment creation
internally, so additional dependencies are typically unnecessary when using
these tools.
