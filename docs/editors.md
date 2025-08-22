# editors

This document will help you to identify a text editor or IDE to use for a more
complete development experience. After reading this document, consider looking
at these documents for next steps in setting up a complete local development
environment:

- [language tooling](#./language-tooling.md) and
- [package managers](#./package-managers.md)

As we saw in this course, Jupyter notebooks provide an excellent interface for
prototyping and sharing code. However, notebooks become an impedance when
working on larger software projects or when trying to integrate more tools into
your workflow. If you find yourself reaching the limits of Jupyter notebooks,
the next step will be to find a text editor that you can write full-fledged
programs in.

## Goals when picking a text editor

Based on my experience, there are a few goals you will want to satisfy when
using a text editor. A non-exhaustive list of things to consider are

* Support for editing documents of several filetypes. In particular, you will
  probably want good integration (or no integration) for writing:

  - markdown (*.md)
  - python (*.py)
  - C/C++ (*.c, *.cc or *.c++)
  - latex and latex bib files (*.tex, *.bib)
  - plain text (*.txt, typically)
  - tabular data (*.txt, *.csv, *.tsv)

* A terminal for running programs and exposing other utilities.

* A debugger for stepping through code as it executes to troubleshoot errors.

* The ability to render Jupyter notebooks.

* Some form of [version control](#./vsc.md) integration.

* Added tooling such as formatters, linters, and autocomplete to make writing
  code easier; collectively, these could be called [language tooling](#./language-tooling.md).

* Remote development support, primarily for connecting to things like clusters.

* Responsiveness and performance. Editing documents is no fun when there is lag
  between what you type and what the display renders.

You may have other priorities, but these cover many of the primary concerns
people try to address when selecting a text editor.

## My recommendations

I am recommending two free and open source editors I actually use and one paid
editor that it would be criminal not to acknowledge here for completeness. In
order of their approximate utilization, the editors I (dan) most highly
recommend are

1. [VS Code](https://code.visualstudio.com/): An open source editor developed by
   Microsoft, VS Code has extensive feature integration to address every feature
   in the _desi derata_ above. It is not the most performant editor, but that
   should not be the highest priority when getting started anyway. In particular,
   VS Code offers great support for remote development in virtual environments
   (primarily containers) and on HPCs via ssh tunneling. Microsoft has committed
   a lot of developer time to making data science work nicely in VS Code, and as
   much if not more time making the debugger a world class debugger. No matter
   what editor you use, you may find VS Code worth having for these two features
   alone.

2. [PyCharm](https://www.jetbrains.com/pycharm/): This is a commercial editor
   developed by JetBrains, and is extremely popular among its user base. There
   is a free tier for trying the product with limited features and a paid tier with
   more support for users who enjoy the product. Students may also be eligible for
   free access to the paid tier as part of various education incentives.
   **Disclaimer: I have never personally used PyCharm, but it has too many users
   not to acknowledge as a popular choice**.

3. Vim (and specifically [Neovim](https://neovim.io/) for local development):
   Vim is a 30+ year old editor that can be widely found in almost any
   Linux-based cloud server or HPC. It does not ship many features out of the box,
   but it is a nice editor to be familiar with if you plan to do work on an HPC or
   in a cloud. If you decide you like vim enough to use it for plain text editing,
   then you will really enjoy Neovim for its integrations with some of the tools
   mentioned previously. Both Vim and Neovim are entirely free and open source, and
   these are my top choices for any work except for running Jupyter notebooks.

## The most important editors to know about

Without further ado, this is an alphabetical list of editors to consider. These
are separated into full IDEs (all free), paid IDEs, and TUI apps/modal editors
that you will find on an HPC or use for their own sake.

### Full IDEs

* Cursor: A fork of VS Code that provides remarkable support for LLMs. It is
  incredibly popular with new users and folks who want a first class experience
  with LLM code completions. A major benefit of Cursor as a VS Code fork is that
  it shares many of the same features alongside the extensive engineering
  they've done to ship fist class LLM integration.

* GNU Emacs: A free and open source editor with too much historical significance
  not to mention. I've never used it, but some users may find it appealing. It
  runs in its own window typically. As far as I'm aware, you would only use this
  to write python programs. There would not be much support for things like
  Jupyter notebooks in Emacs.

* Jupyter Lab: Technically a full integrated environment, you will probably see
  Jupyter lab on HPCs and in cloud computing environments (Google Cloud
  Platform, Microsoft Azure, Amazon Web Services). It has decent support for
  running Jupyter notebooks purely through a webpage, and allows you to interact
  with a shell in whatever environment it runs in. It's probably not a first
  choice for local development for many people, but some users do like it for
  that purpose.

* RStudio/Posix: R users may like to try developing Python in RStudio. This is
  possible and I've seen others do it, but it is generally easier to pick a
  different tool for better support. Still, RStudio provides the best data
  science interfaces of any development environment on the market.

* VS Code: Microsoft's free, highly extensible editor. It provides first class
  integrations for data science (primarily Jupyter notebooks) and an incredible
  debugger. Its integrations for Git and GitHub are first class as GitHub is a
  Microsoft product. Its support for remote development in devcontainers and on
  HPCs is also something to take seriously if you're doing a lot of work in
  those areas.

* Zed: Zed is a multiplatform editor that is designed to be fast and responsive.
  It is primarily intended as an alternative/competitor to VS Code. One of its
  defining features is its multiplayer support for people to pair program. Note:
  Zed is not officially available on Windows, so it may not be a serious option
  unless you're using Mac or a Linux distribution.

### Paid editors

* Sublime text: Sublime has free and paid tiers. It is a minimalist text editor
  with decent tool support for things like syntax highlighting and "go to
  definition" features. It runs in its own hardware rendered GUI and does text
  editing very well. You may like this option for LaTeX or Typst or Markdown,
  even if it is not your preference for writing code.

* PyCharm: This is a full integrated environment focused mostly on providing a
  good Python development experience. It may support other document formats, but
  the most feature-rich support is going to be around things necessary for
  Python developers. This means markdown and plain text will probably work nicely.

### Text User Interfaces

Apps in this category run primarily in a terminal interface. These are most
commonly found on [HPCs](#./hpc.md), but a growing number of users like them as
their standard text editors. These are all part of a family of _modal editors_
that run purely in terminals.

* Helix: A modernized version of Neovim with more extensibility and great
  multicursor support. Not likely to be installed in an HPC.

* Neovim: A Vim fork with extensive plugin support to get modern features such
  as language servers, formatters, and other nice integrations.

* Vi/Vim: For HPC work specifically, you will almost always find Vi or Vim. If
  you will do HPC work, it is remarkably helpful to have a basic understanding of
  how these two editors work so that you can quickly modify on the server without
  having to copy it to a local device.
