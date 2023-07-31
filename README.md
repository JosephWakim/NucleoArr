# Binding to Sliding Nucleosomes

Linear polymer model with sliding binding sites.

### Setup

1. Clone this repository to your local machine.

```
$ git clone https://github.com/JosephWakim/sliding_nucleosome.git
```

2. Navigate to the repository on your local machine.

```
$ cd sliding_nucleosome
```

3. We manage dependencies with Anaconda. Please install Anaconda before
installing this codebase. Details for installing Anaconda can be found
[here](https://docs.anaconda.com/anaconda/install/). With Anaconda installed,
please run the `make.sh` script to create a new conda environment called `slide`
with all dependencies. Please note that this will overwrite any existing conda
environment called `slide`.


```
$ bash make.sh
```


4. Activate the `slide` conda environment.

```
$ conda activate slide
```


### Theory

For the derivation of the theory underlying the sliding nucleosome and
associated binding models, please visit [this
link](https://www.overleaf.com/read/bykvyyszksfd).


### Examples

Using our theory, we simulate ensembles of linear, euchromatic fibers, and we
evaluate the resulting linker-length distributions.
These examples are implemented in Jupyter notebooks in the `examples/` directory
of this repository.
