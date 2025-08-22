# language tooling

This document explains tools like language servers and plugins that can improve
your experience writing Python in an IDE. If you haven't picked an editor yet,
start in [editors](#./editors.md) and then continue to [package
managers](#./package-managers.md) before reading these docs.

Modern programming languages ship several tools to simplify or improve the
development experience. These include

* Language servers for autocomplete, navigating code, and fixing common bugs.

* Linters to provide syntax highlighting / coloring.

* Formatters to standardize the shape of code (indentation, strings, etc).

* Debuggers for stepping through code as it executes so that you can inspect the
  state of a computation when things aren't going as expected.

This document briefly surveys these tools for Python with suggestions about
which tools are best.

## Language servers

A language server is a tool following a server:client relationship inside your
text editor. At a high level, these tools provide sophisticated navigation and
fixes for common bugs in code. They can aid in finding where pieces of data
(functions, classes, variables) are defined, renaming variables within a
particular scope, and much more.

VS Code offers the most optionality for LSPs, including extensions to use

- Pylance
- Pyright
- Basedpyright

Due to licensing, I believe Basedpyright is the most commonly available language
server in other editors.

## Linters and formatters

The original linters and formatters in Python were tools like `flake8`, `pylint`,
and `black`. Many repositories will still use this family of tools along with a
type checker (see next section). However, a modern replacement is `ruff`, which
is developed by the same group as the `uv` [package
manager](#./package-managers.md).

The ruff formatter and linter is the most popular choice for new projects, and
it has great support in most modern development platforms. Configuration for
`ruff` is as simple as following the [configuring
ruff](https://docs.astral.sh/ruff/configuration/) instructions provided by the
developers.

## Type checkers

Python is a dynamically typed language, meaning that the types are resolved at
runtime. Comparatively, languages like C and C++ and Rust are statically typed
languages, meaning that the types for all variables must be fully resolvable
when reading the files as a developer or when a compiler compiles the code.

Dynamic typing is great for experimentation and quick development, but it has
the unintuitive consequence of making code harder to read and understand at a
glance. To see what I mean, consider this snippet:

```python
def add(a, b):
  return a + b

add("Dan ", "Drennan")    # returns "Dan Drennan"
add(0, 1)                 # returns 1
add(0, 1.5)               # returns 1.5
add([0], [1, 2])          # returns [0, 1, 2]
```

These are all cases where things just work, and highlight the appeal of dynamic
typing. But things don't always "just work", and for less contrived examples it
can be unclear what inputs a function or class actually supports. Type hints
make this more clear to a user. For example, if the `add` function is only
intended for adding integers then that intent could be document by writing the
same function as

```python
def add(a: int, b: int) -> int:
  return a + b
```

Unlike static languages, these types are not enforced at runtime, so they are
unfortunately just guidelines. However, they do allow a developer to tell you
what types of inputs the function is expecting and what types of outputs you
should expect. Types can be strictly enforced using libraries (packages) like
`typeguard`, but more often they are only presented as soft guidelines.

This is all to say that types in Python are mostly type **hints**, and they
serve primarily as guidelines about expected contracts. Type checkers review
your code and try to assist with proving things about the types and behavior of
code. Using the last typed example, if we wrote code calling the add function
as `add(3.0, 2.0)`, a type checker would loudly complain that we had violated
the type hints and we should revisit that code to confirm it is being called
correctly.

The use of type checking in Python is a divided topic, but it is generally
helpful in large code projects. If nothing else, type hints serve as a nice
reminder to you for your own projects about the expectations of an environment.

The traditional tooling providing type checking is `mypy`. But mypy is a little
slow and can be irritating to wait on, rendering it less effective purely
because people skip using it. More recent implementations of type checking
include [pyrefly](https://pyre-check.org/) and [ty](https://docs.astral.sh/ty/),
which are both still in early development.

You may not find these tools helpful early on. If you start developing your own
projects and libraries, however, they may become very useful for that type of
work.

## Debuggers

Take the code snippet we used before and consider now the following function
call:

```python
def add(a: int, b: int) -> int:
  return a + b

add((0, 1, 2), "this is gonna break!")
# TypeError                                 Traceback (most recent call last)
# Cell In[2], line 1
# ----> 1 add((0, 1, 2), "this is gonna break!")
#
# Cell In[1], line 2, in add(a, b)
#       1 def add(a, b):
# ----> 2     return a + b
#
# TypeError: can only concatenate tuple (not "str") to tuple
```

We called this code with two type that cannot be added together, a tuple of
integers and a string. When Python tried to execute that piece of code, it
realized we had tried adding two incompatible variables together and raised an
error telling us that this was not permitted. In a less contrived case this
error message may not be obvious, and debuggers make this stuff easier to
understand.

This is a long winded way of saying that debuggers are super helpful tools and
learning how to use them can be an incredible productivity hack for programming.
VS Code and PyCharm and Zed provide the cleanest experiences for these among the
editors that are mentioned in the [editors](#./editors.md) doc. Other options
are possible in the other editors, but you may resort to print debugging, which
means inserting statements in the code that are aimed only at helping to
understand the debugging process.
