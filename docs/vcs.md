# version control

If you're writing code for projects then I highly recommend learning to use a
version control system. The canonical choice is to learn Git, which is a command
line tool with several GUI-based tools for support (Git Kracken is great). And
Git repositories can be uploaded to public platforms such as GitLab and GitHub.

This is a topic that is nice to learn over time, but language models are
remarkably helpful for learning how to use. The amount of a VCS system that you
need to know to get started working is miniscule in comparison to the depth of
the topic. For example, what I show below is perfect for a solo project in your
academic lab. To contribute to a major open source project would take more than
this, but not much more. To maintain a large open source project, you would want
to know a _lot more_ than what is covered here.

## Git

The quick start for Git specifically is to learn what the following commands do:

```bash
git init              # create a repository
git add <file>.<ext>  # add a file or file changes to version control
git commit            # create a message about the changes being stored
git log               # view a message history for past changes of code
```

There are many great tutorials on how to use Git on YouTube, so I won't
elaborate here. It is also helpful to spend time reading their official
documentation at [git-scm.com](https://git-scm.com/).

## What goes in version control?

Version control systems are designed and optimized for storing source code and
plain text documentation. They can store binary large objects (blobs), such as
images and pdfs and custom data formats, but storing those artifacts in version
control is bad if the objects are changing frequently.

The files you'll typically see in version control for data science projects are:

- code for the project (`*.py`, `*.R`, `*.c`, `*.h`, `*.sh`)
- documentation files (`README`, `LICENSE`, `CONTRIBUTING`, `*.txt`, `*.md`, `*.tex`, `*.bib`, `*.sty`)
- rendering code for project websites (`*.css`, `*.html`)
- project config files (`*.toml`, `*.lock`, `*.json`, `*.yaml`, `Dockerfile`)
- notebooks as documentation (`*.ipynb` or `*.py` conversions thereof)
- tabular data formats (`*.txt`, `*.tsv`, `*.csv`)

Files you don't generally want to see in data science repos include (and should
explicitly ignore in your `.gitignore`):

- binary executables (`*.exe`, `*.dll`, `*.o`, `*.so`, `__pycache__`, or plain files `app`)
- images (`*.png`, `*.svg`, `*.jpg`, `*.tif`)
- rendered or highly structured documents (`*.pdf`, `*.ps`, `*.docx`, `*.xlsx`, `*.pptx`)
- serialized or compressed data (`*.pkl`, `*.npz`, `*.parquet`, `*.hdf5`, `*.nc`)
- proprietary files like cad drawings

There are some exceptions to these rules. For example, images are often placed
in a repo to render in the README file; serialized datasets are sometimes
generated and stored as part of test suites; and rendered documents such as
manuscripts are sometimes included to save readers work.

Generally, if you see file formats like these in a repo you should go inspect
where they are used before executing code from the repo on your machine.

Regarding Microsoft Office formats, those documents are built with their own
version histories, internal storage for images and other formats, and things
that will slow your database down considerably.

## Storing blobs in version control

Git and platforms providing Git integration often support `git-lfs`, a large
file store for some of the binary file formats noted as problematic. The way
these are typically stored is by creating a summary of the bytes in the file and
storing a reference to it outside the rest of the file history. But these can
terribly slow down the performance of checking in code or computing changes from
one epoch of time to another.

## Git alternative: Jujutsu (jj-cli)

A recently developed alternative to Git is [jujutsu (jj)](https://jj-vcs.github.io/jj/latest/).
The main appeal of jj is that it is potentially easier to learn than Git is if
you're starting from scratch. In five years this will either be the new standard
tool used by everyone or it will be an abandoned project. I'd prioritize
learning Git for compatibility with what others are using, but consider using jj
if Git is confusing or tiring to use.

An appealing aspect of the jj design is that they wrote it with friendly
wrappers around Git, meaning you can use this system on your machine but still
interact with a Git source control with ease.

## Guidelines for using version control

Version control systems are made to manage raw source code. This means that it
is generally bad to store in version control

- binary executables
- images in any format
- many file formats (especially things using compressed byte streams)
- jupyter notebooks _after_ they have been run (but GitHub has actually made
  this work fairly nice in the last few years as long as you're not looking at
  file diffs in jupyter notebooks)

The guidelines here are written for Git because it is the most commonly used
tool, it is the tool you're most likely to collaborate with others over, and
jujutsu has an initialization strategy that essentially emulates or wraps git
commands for some of its work. So the remainder of this document is going to
address the Git workflows.

The reason to avoid putting file formats like the ones listed (and anything
related or similar) is primarily because they make huge changes in your code
base and can quickly bloat it, both on the file system and on GitHub. This has
annoying performance implications that you'd really like to avoid, for one, and
introspecting things in Git can become much harder if you're looking through
diffs or version history and you start seeing blobs (binary large objects, which
is how any non-text data gets stored).

The primary tool for ignoring files is a `.gitignore` file, which you can grab a
templated copy of from [github's gitignore
repo](https://github.com/github/gitignore) or generate via a package manager.
This repo's [.gitignore](#../.gitignore) was mostly auto generated by the `uv`
[package manager](#./package-managers.md), for example. You may want to add to a
`.gitignore`, which is best done in the local `.gitignore` for things that are
specific to your current project.

Git also provides a **global** `.gitignore` file in your home directory. I use
this to ignore things that I commonly do in my file system that others don't
need to see. These are things that are idiosyncratic to my workflows
specifically, but no other person should need to know about. For example, I will
very often have a shortcut to a directory of config files or datasets that I'm
running my code against, and I don't want to let Git track those directories. If
I want to share something out of one of those channels then I will put it into a
place where a collaborator would expect to see it and provide documentation
about what the thing is and what it is for.
