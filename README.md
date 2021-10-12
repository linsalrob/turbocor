
# turbocor

This is little program designed to compute (sparse) Pearson correlation matrices as
efficiently as possible.

The main ideas are thus:

1. Pairwise correlations computed using AVX2 and FMA instructions.
2. Multithreading, of course.
3. Save space by discarding correlations with absolute value below some threshold. (Hense, "sparse" correlation matrix.)
4. Use constant space and avoid thread contention and overhead from reallocation by using
thread local buffers which are dumped to a temporary file when full.
5. Find the top-`k` correlations using partial sort taking `O(n^2 + k log(k))`
    time, rather than full sort taking `O(n^2 log(n))` time.

Combined, this lets you compute a correlation network in ~10min and 10GB
memory that would ordinary take a day and hundreds of GBs.

## Usage

There are two subcommand: `compute` computes the sparse correlation matrix, and
`topk` does a partial sort of that matrix to find the top-`k` correlations by
absolute value.

Compute sparse correlation matrix (rounding entries with absolute value below 0.7 to zero). Features
are expected to be in hdf5 format in a dataset determined by the `--dataset` argument.
```
turbocor compute --lower-bound 0.7 --dataset "some_interesting_data" features.h5 correlations.h5
```

Write the top-`k` correlations in comma delimited format to standard output:
```
turbocor topk 1000 correlations.h5 > topk-correlations.csv
```

## Possible improvements

The most obvious way to make this even faster to is to compute pairwise
correlations on a GPU. A less obvious way would be to try to tweak the order
matrix entries are computed in to improve data locality.
