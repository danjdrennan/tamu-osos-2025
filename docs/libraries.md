# libraries

The OSOS workshop gave an introduction to data analysis in Python. Here we
expand on this to mention some of the many packages that are important or useful
to scientific computing and data analysis in Python. These are arranged
into sections. Briefly, these are

* Arrays and linear algebra (numpy and non-ML frameworks)
* Plotting libraries
* Data readers for tabular and nontabular data
* Machine learning frameworks
* JIT compilers and task schedulers
* Testing and interfaces

## Arrays and linear algebra

Arrays implemented in a low level systems language are the foundation to most of
the scientific computing stack in Python. As seen in the workshop, NumPy is at
the heart of most of this development in the Python ecosystem. This section
touches on the non-ML packages that are important to modern development.

* NumPy: Provides `ndarray` types with several `dtype`s for allocating memory.
  In case we did not cover it in much detail, take a look at Numpy's `linalg`
  and `random` submodules. It is also helpful to understand what `ufunc`s are in
  NumPy and how to get approximately similar performance using their
  `np.vectorize` function. That functionality is helpful, but we will see
  several other alternatives if your goal is to make things go faster. For
  tasks like signal processing, the `fft` module is helpful also. The full
  [NumPy documentation](https://numpy.org) is valuable.

* SciPy: This is a close cousin of NumPy, but provides different interfaces in
  some of its submodules. The main additions in SciPy that you could find useful
  are its

  - `stats` module, which provides statistical distributions and a few simple
  univariate/multivariate tests.
  - `integrate` module, which contains ODE solvers.
  - `linalg` module, which provides different implementations of most of NumPy's
    `linalg` module.
  - `sparse` module, which provides sparse variants of some data structures.
  These are handy if you work with datasets that are populated with lots of
  missing data, as they compute faster than storing that data in a dense matrix
  and can save quite a lot of memory.

* NVidia's [CuPy](https://cupy.dev/) provides GPU support for plain linear
  algebra with APIs very similar to the ones used in NumPy and SciPy. Many users
  will get GPU support for their code somewhere else, but this is a nice package
  to know about if you aren't planning to do machine learning.

* The [autograd](https://github.com/HIPS/autograd) library was one of the early
  versions of a good autodiff library in Python. It's lightweight compared to
  full ML frameworks, making it a good library to be aware of if you just want
  numeric differentiation without a full ML framework. It's also a good
  reference implementation to study.

### Accelerating NumPy: Numba and Dask

It is difficult to pick a place to discuss these. NumPy and the packages listed
above give you pretty good performance for computing on large data, but they can
still be slow if you're doing something that requires a for loop or any kind of
asynchrony. Two packages that wrap NumPy to improve performance in this area are
Numba and Dask.

* [Numba](https://numba.pydata.org/) is a JIT compiler built on LLVM, and
  provides support for compiling some programs in NumPy to spend less time
  executing code in Python and more time executing native code. It's an
  interesting project and can be very helpful for staying in Python while also
  squeezing out a touch more performance than you get from NumPy alone.

* [Dask](https://www.dask.org/) targets parallelization in Python. It does this
  by building a graph of your code and determining which parts can run
  concurrently. It then executes that graph with some things conducted in
  parallel, reducing the wall time of computations. It is a highly recommended
  utility for tasks that are easily parallelized.

## Plotting libraries

Plotting is better in Python than many other languages in my opinion. The
flagship library for this is matplotlib, but there are many others worth
investigating if that is not your preferred tool for data visualization. Some of
the noteworthy ones include:

* [arviz](https://www.arviz.org/en/latest/) for plotting traceplots and other
  posterior summaries for Bayesian modeling. That package will point you to
  innumerably many other Bayesian modeling packages, some of which I point out in
  the statistical frameworks section momentarily.

* [plotnine](https://plotnine.org/) is very similar to gg plot, implementing a
  grammar of graphics. If you like that kind of thing then plotnine is a great
  library to consider.

* [seaborn](https://seaborn.pydata.org/) is a statistical plotting library built
  on matplotlib. It is especially useful if you are looking for things like
  kernel density estimators and better boxplots etc than you get out of the box
  in matplotlib.

* [matplotlib](https://matplotlib.org/) is the flagship plotting library, and
  was built to be similar to MatLab's plotting utils. It is fairly intuitive to
  use and makes publishable figures if you are willing to spend time with it.

* [datashader](https://datashader.org/) is a plotting tool using
  [shaders](https://en.wikipedia.org/wiki/Shader). It uses Dask and Numba in the
  backend to accelerate computations, and can facilitate plotting beautiful
  graphics quickly that would otherwise be impossible using something like
  matplotlib.

* Plotly, Altair, and Bokeh are additional libraries that some people consider
  helpful in production environments. The main reason for this is the
  dashboarding capabilities they enable.

## Data readers (tabular and nontabular data)

We touched on Pandas in the OSOS workshop. It and a few other nice libraries are
listed here.

* [Pandas](https://pandas.pydata.org/) is a mature tool for analyzing tabular
  data in Python. At the least, it provides several nice interfaces for reading
  and writing a large range of data formats.

* [Polars](https://pola.rs/) is a multilanguage alternative to Python that is
  written in Rust and boasts impressive performance improvements over most of its
  competitors. It is new compared with Pandas, but is quickly growing in
  popularity. Like Pandas, Polars is primarily targeted to tabular data formats.

* [Xarray](https://xarray.dev/) is a package for reading, writing, and managing
  nontabular data in Python. Technically it handles tabular data as a special
  case, but it really shines for handling nontabular datasets. Its early use
  cases were primarily supporting climate modeling, but recently it has grown in
  popularity for biological datasets. One of the major advantages of xarray is
  its _support for memory mapping large datasets_: By loading an index to a
  dataset and querying it for the data to read into main memory, xarray supports
  handling datasets that would normally exceed the RAM on your computer. This is
  an incredible package for that stuff alone, and I highly recommend checking it
  out if you plan to work with large datasets (GBs or TBs in scale) often.

## Machine learning frameworks

The evolution of machine learning frameworks in Python is interesting. A list of
classical packages for statistical learning as well as the modern deep learning
frameworks are listed here:

* [Scikit-learn](https://scikit-learn.org/stable/) is a classic package
  implementing most of the tools you'd like for linear regression, classification,
  random forests, clustering, and the full range of classical statistical learning
  methods. If you're looking for baselines then Scikit-learn probably contains one
  of the base models you're interested in using. It also provides nice utilities
  for data splitting and loading some historically significant datasets.

* [Keras](https://keras.io/) advertises itself as "deep learning for humans". If
  you want to use neural networks for something but don't want to be bothered
  with the fine details just yet then Keras is probably the package for you to
  start with. Keras is also nice because it has bindings in R, meaning you can
  get a familiar interface in multiple languages using Keras.

* [Tensorflow](https://www.tensorflow.org/) is a multiplatform framework for
  training and evaluating neural networks. It also has a large ecosystem of
  dependencies that are useful, such as
  [tensorflow_probability](https://www.tensorflow.org/probability). The main
  reason to use Tensorflow today would be its multilanguage support, as it has
  bindings in Javascript and a few other languages. But the market for modern ML
  is in PyTorch and JAX, listed next.

* [PyTorch](https://pytorch.org/) is the flagship of deep learning and neural
  networks at the moment, and has been for at least five years. Many of the
  frontier labs developing language models use this for development today, and
  LLMs can generate code in this framework better than any other due to the
  massive troves of training data in the framework. There are several packages
  that extend PyTorch into other use cases than they support out of the box.

* [JAX](https://docs.jax.dev/en/latest/) is the latest ML framework and is
  rapidly growing in users. It is the most similar to NumPy in syntax and
  implementation, and has wonderfully powerful abstractions for distributed
  computing (multi GPU). One of the main features of JAX is its JIT compiler,
  which allows you to write code in Python once and then compile it to native
  code that runs either on CPUs, GPUs, or Google TPUs. In addition, it provides
  simple abstractions for parallelizing and distributing work on or across
  multiple accelerators (GPUs). There is a massive ecosystem built around JAX,
  and that ecosystem is pretty rapidly growing.

## Statistical frameworks

Statistical packages are not as common in Python as they are in R. But there are
a few, and they are worth knowing about.

* The [statsmodels](https://www.statsmodels.org/stable/index.html) package is
  great if you are doing regression modeling in Python and want something similar
  to R's tabular summaries of data fits. Nothing compares to R for linear
  modeling, but statsmodels makes up a lot of ground for the basics. This
  doesn't quite fit anywhere else, so I'm putting it here. It's built on NumPy
  of course. I have found the time series submodule of this package helpful
  personally.

* PyMC3 is a Bayesian modeling library that is quite popular. It is an
  alternative to Stan that is a bit easier to work with in Python specifically.
  Another Python package for Bayesian modeling is Liesel, which is implemented
  on top of JAX. Blackjax is yet another library in the JAX ecosystem for doing
  Bayesian computations.

There are several other packages built on top of one of the ML frameworks. LLMs
are the easiest resource for keeping up with this growing list of packages.

## Tracking software

If you're training models or doing expensive data analyses, you may eventually
find it helpful to use some form of software to help track the hyperparameters
and results of your runs. If you are industry minded at all, it can be helpful
to your CV and application if you've familiarized with at least one of these
tools for model tracking. A few resources for doing that are listed here:

* [Weights and biases](https://wandb.ai/site) is a paid product with a free
  tier. Using a context manager, you can wrap any training run and log its
  events to a weights and biases server for easier model tracking. The results
  of those runs also get stored locally on your machine, and end up in both places
  with a database logging the results for easy querying. They also provide some
  amount of autotuning and grid search to scan over a large family of
  hyperparameters quickly.

* [MLflow](https://mlflow.org/) is an alternative to weights and biases that has
  free and paid tiers. As with `wandb` (above), MLflow provides nice monitoring
  tools that are easy to wrap around your model training runs to get good
  logging and summarization. This group also provides good tooling for deploying
  trained models into a production environment. I think they achieve better
  generality than Weights and Biases in general, as you can use MLflow to monitor
  runs in a large range of modeling packages.

## Configuration parsing and tuning

If you reach a point where your workflow for training a model looks something
like

```bash
python my_model.py --config ./configs/my_model_config.yaml
```

then you may want to try some of the following packages. They take tasks that
are not that difficult to implement yourself and convert them into library calls
so that you aren't doing things like input hashing and validation.

* [Pydantic](https://docs.pydantic.dev/latest/) is an input validation library.
  It helps you to express the inputs you expect through an elaborate type
  system, and automatically handles validation of that data for you. This helps
  when you're setting things up like `learning_rate` as a hyperparameter to a
  model and you want to be sure the provided parameter is valid before using it in
  a model (learning rates tend to be in the unit interval [0, 1]).

* [Click](https://click.palletsprojects.com/en/stable/) is a drop-in replacement
  for Python's `argparse` module from the standard library. It makes it easy to
  decorate a function to build an argument parser automatically and with pretty
  good documentation. Whether you use this or argparse is up to preference, but
  many people enjoy this package as a simplified mechanism for getting decent
  command line interfaces.

* [Optuna](https://optuna.org/) is a decent library to help you automate grid
  searches over hyperparameters in a model.

## Software testing

A nice way to document the behavior of your code is using various forms of
software testing. This won't necessarily help you publish papers or finish
projects faster, but it can help you to gain confidence in the correctness of
your implementations. It can also give you a detection layer for major changes
in your dependencies (which you will have a lot of, as you can see from here).

* [Pytest](https://pypi.org/project/pytest/) is the de facto standard for
  writing unit tests and more elaborate test suites. It can discover and run a
  large family of tests for you automatically. It also integrates with extension
  packs that can help you gauge the coverage of your tests or to run performance
  benchmarks as part of your test suite. However, it is often more than
  sufficient to just test that your code does what you think it does and to
  document how you handle bugs. The other things are interesting to measure but
  don't have nearly the impact of unit tests.

* [Hypothesis](https://hypothesis.readthedocs.io/en/latest/) is a tool for
  writing property-based tests in Python. This is great if you want to gain
  confidence in the robustness of your implementation for something.

## Writing low level extensions

Occasionally (but increasingly rarely, more and more of the code you call in
Python binds to native code or can JIT compile to native code), you may have a
problem that needs to run fast and which is not easily expressed in Python or
one of its accelerated languages. These sorts of extensions are called through
_foreign function interfaces_ (FFIs). If you need to write a low level extension
for your application, then the general steps to writing the extension are

- write the extension in your low level language
- write a build system for compiling that low level code to an executable binary
- write Python bindings that know how to call the low level code
- if packaging code, then write builder code so that Python can call the
  compiled code

If you are going this route then there are several options for writing Python
bindings for your low level code. This is a non-exhaustive listing with some
examples of how to do that.

* Use Python's builtin tools for exposing your extensions to the CPython
  interpreter, which they present in their
  [docs](https://docs.python.org/3/extending/extending.html).

* Use [Pybind11](https://pybind11.readthedocs.io/en/stable/basics.html), which
  exposes a cleaner interface for writing C++11 code and binding it in Python.
  The maintainers go further than Python and provide NumPy headers and things that
  make it easier to interact with frameworks that you're probably planning to
  interact with from your low level code anyway. **This is likely the preferred
  method if you're just writing something to extend NumPy and base Python**.

* PyTorch provides its own
  [mechanism](https://docs.pytorch.org/tutorials/advanced/cpp_extension.html)
  for extending PyTorch specifically. This is built on top of Pybind11, but
  PyTorch gives instructions for how to extend PyTorch code in their docs.

* JAX provides similar [docs](https://docs.jax.dev/en/latest/ffi.html) for
  building FFIs in their code.

* For Rust-based extensions to Python, or Rust-Python interopability more
  broadly, there is the [PyO3](https://github.com/PyO3/pyo3) package. It uses
  [maturin](https://github.com/PyO3/maturin) for writing bindings, and is
  developed and maintained by the same PyO3 org.

A good use case for writing custom C/C++/Rust code to bind into Python is if you
need to write a bespoke sorting algorithm or tree-based data structure yourself.
Another reason you may want those bindings is to use a low level implementation
of something like an MCMC sampler that was written in C or C++, but to call it
from Python. There's tons of code that fits this use case where there would
still be good opportunity for binding something in Python that hasn't been
implemented yet.

A growing alternative to all of these is to use the
[Mojo](https://www.modular.com/mojo) language, which gives you hardware level
control of something within a Python syntax. This is one of the most recent
developments for this kind of programming and lowering, and would be a better
starting point if C and C++ are unfamiliar but you need performance.
