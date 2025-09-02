"""numpy: an introduction to numpy, scipy, and matplotlib"""

import time
import timeit
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

# Numpy is a numerical python library implementing almost everything you would want to
# use from classical numerical linear algebra. It has a sister package that is closely
# integrated in many submodules, and here we will cover a subset of both. The plan is to
# briefly go over different data types, data layouts (arrays), functions, linear
# algebra, and random number generation. There are many other libraries in Python for
# doing deep learning which use quite similar APIs. Almost every concept we discuss in
# this module would carry over to those frameworks. Among those frameworks, PyTorch is
# the most popular deep learning framework in use today, but JAX is a terrific ecosystem
# with much tighter coupling to Numpy.

# ------------------------------------------------------------------------------------
#                                      DATA TYPES
#                             * ints
#                             * uints
#                             * floats
#                             * complex
#                             * object
#                             * type promotion
# ------------------------------------------------------------------------------------

# Like Python provides several builtin types to represent numerical data, Numpy provides
# many of its own types. A key difference is that all of numpy's types are fixed width,
# meaning that they can overflow. For example:
i64 = np.array(1)
print(i64.dtype)

# We can also ask for unsigned versions of integers, or cast data from one type into
# another (dangerous if you aren't positive of domain restrictions). For example:
u64 = i64.astype(np.uint64)
print(u64)
print(f"{i64-2=}, {u64-2=}")
print(f"{u64 - 2 == 2**64 - 1}")
print(f"{(u64 - 2).astype(np.int64)}")

# The signed integer uses its first bit to indicate the sign of the number and its
# remaining 63 bits to represent the integer itself. So the range of representable
# values are -2^{63} <= i63 < 2^{63}. The unsigned bit does not reserve that first
# position, so we have 0 <= u64 < 2^{64}. And as we see, integers can be recast between
# types to be interpreted differently.

# Because Numpy stores integers in fixed space, we can also overflow them with undefined
# behavior. Consider now what happens when we shift the bit left by several positions:
print(i64 << 63)
print(i64 << 65)

# You should practically never do any of these things, but it is useful to compare this
# behavior with default Python's corresponding behavior.

# Floating point types are a similar story. By default, Numpy stores everything in
# float64. It also aggressively promotes things internally to float64. So if we just
# create a float
f64 = np.array(1.0)
print(f64, f64.dtype)

# we see it is using 64 bits. We could just as well store things in float32, in which
# case we have
a = np.array(1.0, dtype=np.float32)
b = np.array(2.0, dtype=np.float32)
c = a + b
d = a * b
print(a, b, c, d)
print(a.dtype, b.dtype, c.dtype, d.dtype)

# Numpy also supports complex numbers. Unless you're doing signal processing you
# probably won't find much need for these. But some of you may want to use FFTs in your
# work in which case you will want to know what these are and how they work:
c64 = np.array(1.0 + 0.0j)
print(c64, c64.dtype)

# The full list of numpy's dtypes can be found at
#
#    [Numpy Dtypes](https://numpy.org/devdocs/user/basics.types.html)
#

# A last note: floats are imprecise approximations of the real number line. So we have
print(np.array(0.1) + np.array(0.2))

# This is a consequence of implementing floats using IEEE754, which is the floating
# point standard that basically everyone uses because it's thoroughly baked into the
# hardware we compute on. Whereas in integers we can always make exact comparisons like
print(i64 == u64)

# in floating point, we have to instead ask 'are these two things close'?
a = np.array(0.1)
b = np.array(0.2)
c = np.array(0.3)
print(np.isclose(a + b, c))

# ------------------------------------------------------------------------------------
#                                     DATA LAYOUTS
#                             * scalars
#                             * vectors
#                             * matrices
#                             * tensors
#                             * reshaping
#                             * broadcasting
# ------------------------------------------------------------------------------------

# But the reason we use Numpy isn't really to work on scalars. It's because we want to
# use container types like n-dimensional arrays. We use numpy when we want to do matrix
# math, or just to make some analysis in Python go faster than default Python.

# A good starter is to look at the arange object. Python has `range(start, stop, step)`
# objects. Numpy has `arange(start, stop, step)` objects. Note in both cases that we
# use half open intervals [start, stop).
arange = np.arange(10)
print(arange)

# These arrays are zero-indexed. The first element is accessed at arange[0], like:
print(arange[0])

# Unlike Python's range object, our numpy arange supports instantiation by float. So the
# following are equivalent:
aranged = np.arange(10.0)
arange_double = np.arange(10).astype(np.float64)
print(aranged)
print(arange_double)

# We compared the last two arrays by printing and looking at them. But we can actually
# use comparison operators for the same purpose:
print(aranged < arange_double)
print(aranged == arange_double)
print(aranged > arange_double)

# What happened here? Numpy does comparisons elementwise by _broadcasting_. We will
# discuss broadcasting more momentarily. If we wanted to know if all of the elements
# were the same, we could wrap the last comparison like so
print(np.all(aranged == arange_double))

# Or if we wondered if any of the elements matched, we could instead ask
print(np.any(aranged == arange_double))

# These comparisons may take some practice to get used to, but they become intuitive
# over time I promise.

# Another thing to be cognizant of is type promotion. Numpy _really likes to work in
# float64_. And any time we combine data from different types, Numpy will promote to the
# largest type necessary to store the results.
arangef = np.arange(10.0, dtype=np.float32)
print((arangef + arangef).dtype)
print((arangef + aranged).dtype)

# A last thing we want to discuss before looking at larger array types is broadcasting.
# We saw with comparisons that numpy does elementwise comparison. The same is true if we
# compare our scalars with our vectors.
i64 = np.array(5)
arange = np.arange(10)
print(arange == i64)
print(np.all(arange == i64))
print(np.any(arange == i64))

# Or if we want to know where two things are equal, we can just ask:
print(np.where(arange == i64))

# This returns an index of all of the places where two things are equal. Often, however
# we want to know this so that we can replace some set of values with another set of
# values. For example, we are using `np.where` to replace writing a code snippet like
for i in range(10):
    if arange[i] == i64:
        arange[i] = 1
    else:
        arange[i] = 0
    # or arange[i] = 1 if arange[i] == 1 else 0

# Note the last type was a tuple. But normally when we use the np.where function, we do
# so because we want to conditionally insert values into the array. The following are
# equivalent:
print(np.where(arange == i64, 1, 0))

# Note: we have now permanently altered the data in `arange`.
print(arange)

# Broadcasting:
print(f"{arange + i64=}")
print(f"{arange * i64=}")

# Arrays:
# We can reshape data to do some neat stuff. Consider
arange = np.arange(10.0).reshape(2, 5)
print(arange)

# Or, more simply,
arange = np.arange(10.0).reshape(2, -1)
#      = np.arange(10.0).reshape(-1, 2)         # by symmetry
print(arange)

# We can make the arrays longer and reshape them into any configuration we like:
arange = np.arange(2.0 * 3.0 * 5.0 * 7.0).reshape(2, 3, 5, 7)
print(arange)

# Arrays come with metadata: We can query the amount of storage used for the array, its
# shape, and the dtype (as we saw with scalars):
print(arange.nbytes, arange.shape, arange.dtype)

# Data can be reshaped in whatever way we want, and again we can query the new arrays
# to learn what the shape and data types are. In most cases it is probably clear from
# context what we want to use or do
arange_reshaped = arange.reshape(5, 2, 3, 7)
print(arange_reshaped.nbytes, arange_reshaped.shape, arange_reshaped.dtype)
print(arange_reshaped)

# Finally, we can take subsets of data in the same way as we slice any other object in
# Python; some examples:
arange = np.arange(10.0)
print(f"{arange[3:6]=}")
print(f"{arange[:4]=}")
print(f"{arange[7:]=}")
print(f"{arange[-4:]=}")

# There are several ways in which we may want to create new arrays aside from the ones
# shown. Some of these are with pseudo-random numbers, which we'll get to in a moment.
# But there are several deterministic sequences we may want to create, like
print(np.empty((3, 3)))
print(np.zeros((3, 3)))
print(np.ones((3, 3)))

# Or variations thereof. Related to an arange, we may want to construct something like
# an array of floats. Here are a few ways we can do that:
print(np.arange(0.0, 31.0, 2))
print(np.arange(0.0, 1.01, 0.01))
print(np.linspace(0, 1, 100))

# Another crucial part of numpy is the ability to fuse multiple data. For example, we
# may want to concatenate data in various ways. Here are a few examples:
x = np.linspace(0, 1, 20)
concat = np.concatenate([x, x])
print(concat.shape)

vstack = np.stack([x, x], axis=0)
print(vstack.shape)

hstack = np.stack([x, x], axis=1)
print(hstack.shape)

# Or use these, but it's simpler to remember one function than it is to remember two or
# threee, so I prefer stack with an argument about how I want to stack things.
# vstack = np.vstack([x, x])
# hstack = np.hstack([x[:, None], x[:, None]])

# ------------------------------------------------------------------------------------
#                                    RANDOM NUMBERS
# ------------------------------------------------------------------------------------

# We present this module next because it is imperative to generating data in the
# subsequent sections, so that we can work on less contrived examples than the ones we
# have already seen.

# We don't have to set random seeds, but it makes the analysis reproducible if we
# implement something and want it to run deterministically. Everything we explore today
# will be seeded. The right practice is to set seeds once at the start of a program and
# then execute the program sequentially from start to finish. You will notice if you run
# cells multiple times after setting the seed that the results become apparently random.
# This is one of the primary reasons to strive for running stuff in scripts rather than
# jupyter notebooks. That is a point that will be difficult to get across by the end of
# the workshop, but it is one to consider if you continue to work in Python after this
# workshop. For today, however, we are going to work in Jupyter notebooks because they
# are interactive and nice for exploratory work.

# The seed itself could be lots of things, including an empty argument. I always start
# with a concrete value... Most of my test scripts start with seed 0. If you want to
# have random but reproducible behavior, a better seed is to just grab the system time
# in nanoseconds at program startup, like this:
seed = time.time_ns()
print(f"{seed = }, (in hex: 0x{seed:x})")
rng = np.random.default_rng(seed)

# If we save that seed someplace then we can always reconstruct the program's execution
# using the seeded value. This is *immensely* helpful for debugging programs that rely
# on random number generation!

# The random number generator supports generating all kinds of different distributions.
# Here we will focus on the most commonly chosen ones:
normal_rvs = rng.normal(loc=0.0, scale=1.0, size=(500,))
uniform_rvs = rng.uniform(low=0.0, high=1.0, size=(500,))
gamma_rvs = rng.gamma(shape=5.0, scale=3.0, size=(500,))

if False:
    fig, ax = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")
    ax[0].hist(normal_rvs, density=True, edgecolor="C0", facecolor="lightblue")
    ax[1].hist(uniform_rvs, density=True, edgecolor="C0", facecolor="lightblue")
    ax[2].hist(gamma_rvs, density=True, edgecolor="C0", facecolor="lightblue")
    plt.show()

# There are also great helpers for randomly permuting data, like so:
normal_rvs_perm = rng.permutation(normal_rvs, axis=0)

# Or generating an index by which to reindex all of the data:
permutation = rng.permutation(500)
normal_rvs[permutation]  # Can do this for all three arrays now.

# We could use a permutation and then slice into the array to obtain samples from it,
# but there is a more declarative approach to sampling for arrays:
sample_idx = rng.choice(500, size=50, replace=False)
print(sample_idx)

# Or just to sample directly:
normal_rvs_sample = rng.choice(normal_rvs, size=5000, replace=True)

if False:
    plt.plot(normal_rvs, ".")
    plt.plot(normal_rvs_sample, ".")
    plt.show()

# ------------------------------------------------------------------------------------
#                                      FUNCTIONS
# ------------------------------------------------------------------------------------

# We took the last excursion sampling so that we could explore functions in a less
# trivial manner. Let's generate the normal random variables again, except this time
# let's create multiple dimensions.
x = rng.normal(size=(100, 30, 5))

# To start, some functions are bound to the variables we are working with. For example,
# we can calculate the mean, variance (var), standard deviation (std), minimum (min),
# maximum (max), index of the min (argmin), and index of the max (argmax) using those
# functions bound to the numpy namespace _or_ bound to the array objects themselves.
print(f"{np.mean(x)=:.5f} == {x.mean()=:.5f}")
print(f"{np.var(x)=:.5f} == {x.var()=:.5f}")
print(f"{np.std(x)=:.5f} == {x.std()=:.5f}")
print(f"{np.min(x)=:.5f} == {x.min()=:.5f}")
print(f"{np.max(x)=:.5f} == {x.max()=:.5f}")
print(f"{np.argmin(x)=:.5f} == {x.argmin()=:.5f}")
print(f"{np.argmax(x)=:.5f} == {x.argmax()=:.5f}")


# We can also apply these functions over specific axes. This operation will reduce the
# shape of the array along that dimension, which is avoidable if we use `keepdims=True`.
# Often that is unnecessary.
print(f"Calculating means for {x.shape = } along different axes")
print(
    f"{np.mean(x, axis=0).shape=},\t{np.mean(x, axis=1).shape=},\t{np.mean(x, axis=2).shape=}"
)

# But statistics aren't the only functions we may want to compute. Here is a plot of
# sines, cosines, polynomials, and some of the other things you may generally want.

x = np.linspace(1e-16, 8.0, 100)
functions = dict(
    sines=np.sin(x),
    cosines=np.cos(x),
    tangents=np.tan(x),
    squares=np.square(x),
    cubes=x**3,
    sqrts=np.sqrt(x),
    logs=np.log(x),
    exps=np.exp(x),
)
if False:
    fig, ax = plt.subplots(2, 4, figsize=(8, 4), layout="constrained", sharex=True)
    ax_flattened = ax.flatten()

    for i, (fn, fn_vals) in enumerate(functions.items()):
        ax_flattened[i].plot(x, fn_vals)
        ax_flattened[i].set_title(fn)
    fig.suptitle("A Sample of Numpy functions to calculate")
    plt.show()

# Of course, all of the operators you'd expect to work on arrays do as well.
x = np.linspace(0, 1, 5)
y = np.sin(x)
print(f"all the combos: {np.sqrt(x) + 5 * y**2 / 3 + 2}")

# ------------------------------------------------------------------------------------
#                                    LINEAR ALGEBRA
# ------------------------------------------------------------------------------------

# Linear algebra is one of the core components of much of the work we do in statistics
# and data science. This is a subset of the many linear algebra things we can do.

# Dot products:
u = rng.normal(size=(30,))
print(f"{np.dot(u, u):.4f}")

# Solving *dense* linear systems
A = np.array(
    [
        [0, 1, 2],
        [1, 4, 2],
        [2, 1, 1],
    ],
)
x = np.array([1, 2, 3])
b = np.dot(A, x)

print(f"{A=}")
print(f"{x=}")
print(f"{b=}")

x_solved = np.linalg.solve(A, b)

print(f"||x_solved - x|| = {np.linalg.norm(x_solved - x):.4f}")

# But we should generally try to solve problems by using knowledge of the problem at
# hand. Some of you will have problems where the matrix has special structure (whether
# you are initially aware of this or not). When such structure does exist, you can often
# exploit it in the solver you are using. For example, here is a circulant embeddings
# matrix. These have very efficient solutions if you know that this structure exists in
# your problem.
if False:
    A = linalg.circulant(np.arange(500))
    x = rng.normal(size=A.shape[0])
    b = np.dot(A, x)

    std_solve = timeit.repeat(
        lambda: linalg.solve(A, b),
        repeat=10,
        number=100,
    )

    circ_solve = timeit.repeat(
        lambda: linalg.solve_circulant(A[:, 0], b),
        repeat=10,
        number=100,
    )

    print(f" std_solve: {np.mean(std_solve):.4f} +/- {np.std(std_solve):.4f}")
    print(f"circ_solve: {np.mean(circ_solve):.4f} +/- {np.std(circ_solve):.4f}")

# Banded matrices are similar but not shown here for the sake of time.

# There are several decompositions that are also supported in np.linalg and scipy.linalg
# (imported as `from scipy import linalg`). Some of these that are common to least
# squares problems are shown in `./ols.py`. Others we don't cover are the SVD and
# eigen decompositions. These would be most relevant to PCA. However, good
# implementations of these are typically provided in scikit-learn also via
# `sklearn.decomposition`. Consider having a look at those if you are planning to do PCA
# in your research.
