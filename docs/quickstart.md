# Quickstart

This quickstart guide has instructions for setting up a Python dev environment
in the configuration that most people new to Python follow. It chooses

- An editor (VS Code; see [editors](./editors.md) for alternatives)
- A package manager (uv; see [package managers](./package-managers.md) for alternatives)
- Setting up Git and GitHub for version controlling software
- Language servers and tooling (through VS Code community extensions; most editors offer alternatives)

The complete setup could be done in just a few minutes, but plan on it taking
30-60 minutes total.

Following the initial setup, we walk through how to create a new project and
create an environment in it.

The instructions are for getting set up using a Ubuntu machine. The steps are
very similar on MacOS and Windows. Windows users are recommended to install VS
Code in Windows and to use [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
to emulate a Linux environment for Git and Python.

<details>
  <summary>
    <strong>Installing WSL on Windows11 (expand this)</strong>
  </summary>
  The quickest way to install WSL is to open PowerShell in **administrator**
  mode and then run the following commands (note this chooses ubuntu for setup):

  ```powershell
  wsl --install                 # installs wsl executable
  wsl --set-default-version 2   # sets the default version of wsl to wsl2 (may be deprecated with wsl2 as the default now)
  wsl.exe --install Ubuntu      # installs ubuntu as the distribution to run
  wsl.exe --set-default Ubuntu  # sets ubuntu as default, in case other distros are provided
  wsl.exe                       # starts the wsl environment in ubuntu
  ```

  The next step installing VS Code will be done in your normal Windows
  environment. Extensions should be installed in the WSL environment (see steps
  there). The Git and uv installations can be done in WSL2 following the Linux
  guide.
</details>


## Editor setup

Download [VS Code](https://code.visualstudio.com/) following the instructions on
their website and using your preferred installation method. On Ubuntu you can
install VS Code from the Snap registry using `sudo snap install code --classic`;
note this requires root privileges, but is standard.

### Installing extensions

Out of the box VS Code is just an extensible text editor. Extensions make it
useful for Python development (and other languages if you end up needing that).
To install extensions, click `File -> Preferences -> Extensions`. This will open
a searchable menu from which you can install extensions.

I use the following extensions in my environment, and recommend them as a
starting point.

```extensions
Ruff
Basedpyright
Debugpy
Python
Vscode-python-envs
Jupyter
```

## [Optional] Install Git and configure GitHub

This step is not required if you don't want to version control your code.
However, it is highly recommended to use Git for version control on your
machine. Using GitHub is strictly optional, but often recommended because it
provides a cloud-based backup of your code.

<details>
  <summary>
    <strong>
      Do I have to share my code pubicly if I use GitHub?
    </strong>
  </summary>
  No! GitHub supports public and private repositories. Private repositories have
  limits on the number of collaborators to discourage unrestricted commercial
  use. Additionally, a private repository can be converted to public at any time.
  Converting a repository from public to private is not always possible, so be
  conscientious of this decision when creating a repository.

  Note also that most academics can obtain an education license to GitHub that
  affords most of the benefits of a full commercial license (including many more
  collaborators on private code). Interested PIs can also license GitHub for
  their research groups.
</details>

Git can be installed using a package manager with one of the following commands
in an appropriate terminal/shell:

```bash
brew install git        # macos
sudo apt install git    # debian/ubuntu-based linux (and in wsl)
winget install git      # windows (if winget is installed)
choco install git       # windows (if choco is installed)
```

Alternatively, Git can be installed from their [website](https://git-scm.com/downloads).

> [!NOTE]
> Git is a command line application. Graphical applications and utilities are
> available in VS Code and most editors for users who prefer that workflow.
> Several YouTube videos explain what it is and how to use it, and these
> [materials](https://git-scm.com/doc) are a wonderful resource for a complete
> beginner. See also [our guide](./vcs.md) for some details and templates to
> download.

## Install uv

This is going to cover two steps. First, we are going to install uv. Then we are
going to install Python using uv if it is required. Some notes before the
instructions:

- **Windows users**: Follow the instructions for Linux if you are using WSL.
  Otherwise follow the instructions in Windows.
- **The Python step**: Python refers to several versions of the same language.
  Generally, you want to install version 3.10+. Version 3.13 is the current
  version, and version 3.14 is scheduled to release in October 2025.

The full details of the uv and Python installations may be found in Astral's
docs at:

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Install [Python](https://docs.astral.sh/uv/guides/install-python/) using uv

For Linux/Mac users who trust my instructions, the gist is to run the following
commands in a shell:

```bash
# Install uv (pick one of )
curl -LsSf https://astral.sh/uv/install.sh | sh   # download via curl and install in one line, OR
wget -qO- https://astral.sh/uv/install.sh | sh    # install using wget if curl doesn't work
pipx install uv                                   # works if you have a managed system version of python on ubuntu

# Install Python using uv
uv python install                 # gets latest python (version 3.13)
uv python install 3.1x            # install python3.1x (x = 0,1,2,3)
```

## Creating our first Python project

At this point, you should have the complete substrate to do quite a lot of
Python development on your computer. Great job! The next steps are going to
create our first project.

You are free to do this using a file explorer, but the instructions are most
easily expressed using shell commands. If you'd like to follow along in a Unix
shell, open your terminal and try the following steps:

```bash
cd ~                      # [optional] go to your home directory to start
mkdir projects            # [optional] I like to put code in a projects
cd projects               # folder within the home directory rather than the home directory

mkdir first-project       # creates a directory (folder) for the new project
cd first-project          # navigate into the directory/folder just created

uv init                   # initializes a project in uv [creates many files]
uv venv                   # creates a virtual environment in .venv
```

The first three steps were entirely optional. After the optional steps,we (i)
created a folder `first-project` to develop our code in, (ii) went to that
directory, (iii) created a project, and (iv) created a virtual environment. In a
Unix shell, `ls -laF` will show you a file tree that looks something like the
following:

```tree
.
├── .git
├── .gitignore
├── .python-version
├── .venv
├── README.md
├── main.py
└── pyproject.toml
```

Your view may look slightly different from this if you don't have Git installed.
If you do have Git then your view should be essentially identical to mine. We
will discuss the contents of these files in more detail once we've completed
setup.

### Opening the project in VS Code

We will discuss these files more in a moment, but let's start by opening the
project in VS Code.

```bash
code .                  # open the project in VS Code
```

Now you can view the entire project in VS Code. In the future you may prefer to
open the project in VS Code without using a shell. If you're in VS Code, one of
the ways to open a project is to use `File -> Open Folder ->` and use a
graphical menu to navigate to your project.

### Adding dependencies to our Python environment

At this point we should be in VS Code and looking at our project, which is
located in `~/projects/first-project/` if you've followed all of the
instructions from the previous steps.

VS Code provides a terminal (`File -> Terminal -> New Terminal`) that you may
use instead of the terminal that we used for all of the previous setup. But any
management of Python, using uv or another tool, is going to require some amount
of using a shell.

Most data science projects are going to install the following dependencies, so
let's go ahead and add them. This is a good point to inspect the
`pyproject.toml` file and `uv.lock` file before and after running the next
commands, as uv is going to automatically write content to both.

```bash
uv add numpy pandas matplotlib  # add dependencies to project
source .venv/bin/activate       # activates the virtual environment
```

We can test that everything is working as expected by running the following
command in a shell:

```bash
# Expected output: some number close to zero in size
uv run python -c "import numpy as np; print(np.arange(10).sum())"
```

This last command ran a Python program equivalent to this:

```python
import numpy as np

x = np.arange(10)     # [0, 1, 2, ..., 9]
sum_x = np.sum(x)     # 9 * (9 + 1) / 2 = 10 * (10 - 1) / 2
print(mean_x)         # Expect: 45
```

Hopefully this runs for you and you get the result `45` in a terminal. If it has
not run then there is debugging ahead of you. If there is a bug, I recommend
copying this page of documentation into an LLM and asking it for debugging
support.

### File formats

Recall the file tree we saw earlier:

```tree
.
├── .git
├── .gitignore
├── .python-version
├── .venv
├── README.md
├── main.py
└── pyproject.toml
```

All of these files were auto generated for us when we used `uv init` and `uv
venv` to initialize our project. Some of these files are fully managed by `uv`,
so I want to address them briefly first. Then we'll talk about other parts of
the repo.

#### uv managed files

`.python-version`:
The first file I want to talk about briefly is `.python-version`. This file
tells uv which version of Python we are using for the current project. Since uv
can manage multiple versions of Python simultaneously, we need this file so that
uv can resolve the dependencies correctly.

`pyproject.toml`:
If you monitored `pyproject.toml` and `uv.lock` when writing the command `uv add
numpy pandas matplotlib`, then you should have noticed that the files expanded
with new data. The `pyproject.toml` file inserted specific versions of each
package into a section called `dependencies`, and now looks something like

```pyproject.toml
dependencies = [
  "numpy",
  "pandas",
  "matplotlib",
]
```

You can modify parts of this file by hand, but sections like dependencies should
be left to uv to manage for you most of the time.

`uv.lock`:
The `uv.lock` file contains less readable information, but is equally important.
The `pyproject.toml` file is a human-readable table of all of our
`first-package`'s requirements. The lock file is less human readable, but
creates a precise specification of each dependency we require to rebuild this
package. As long as the package index is available, we can rebuild this project
exactly from the state that is provided.

#### git managed files

If you aren't using Git or if you don't want to use Git for the project just
created, you can remove the Git pieces of this repository using `rm -rf .git
.gitignore`.

The `.git` file in our tree, if it exists, contains a fairly complicated tree
with what will become a complete history of our project. The only times I
interact with this directory are to delete it for projects where I know I don't
want or need version control. Typically these are very ephemeral things I'm
testing somewhere.

The `.gitignore` file tells Git what files not to include in a project. `uv` has
a pretty good template for what to ignore already, but you can modify this to
add files. A useful addition if it's not there is to ignore `.DS_Store`, which
is a piece of metadata from MacOS's filesystem that should never appear in a Git
repository. Another useful ignore is `__pycache__`, which ignores bytecode and
artifacts generated by Python when running code. There are great templates for
things to add to this file online, including
[github's gitignore repo](https://github.com/github/gitignore) from GitHub.

#### README files

Now let's conclude with the files we actually touch. The `README.md` file serves
as an abstract for the project. The easiest way to learn what goes in one of
these is to read other projects on GitHub. Google has a few great repositories
to use as case studies for this purpose. At the very least, a README should tell
the reader

- What the project is
- Who the maintainers are
- How to build/install the project
- Something about the licensing model
- A short demo of how to use the project code as intended

Other things you'll commonly see include

- Citations that can be copy/pasted into a LaTeX bibfile
- Contributing guidelines (see Scipy for example)
- License information
- Instructions for contributing to the project either by donation, development,
  or other means that the maintainers provide

#### Source code

The last bits are python files. These take many forms:

- Scripts in the root directory, like `main.py` in the tree above
- Library code under `src/first_project`
- Test code (code that tests our code) under `tests/test_project_file.py`

The way we used uv to initialize our project did not configure a library as part
of the package. There is a switch to generate
