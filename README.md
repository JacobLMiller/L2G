# LGS graph drawing algorithm

Repository accompanying GD2023 submission: https://doi.org/10.1007/978-3-031-49272-3_18

Required packages are listed in requirements.txt. You can install all with 

```pip install -r requirements.txt```

## Compile cpython
You'll first need to compile the L2G optimization script on your machine. Once cpython is installed, you can do this by 

`python setup.py build_ext --inplace` 

If there are no errors, it should be good to go. 

## Usage

```
Usage: layout.py [-k K] [--alpha ALPHA] [--epsilon EPSILON] [--max_iter MAX_ITER]  [--output OUTPUT] input_graph

positional arguments: 
    input_graph         Input graph to read from. 

optional arguments:
options:
  -h, --help            show this help message and exit
  --k K, -k K           Number of most-connected neighbors to find
  --alpha ALPHA, -a ALPHA
                        Number of powers to consider in adjacency matrix
  --epsilon EPSILON, -e EPSILON
                        Threshold for convergence.
  --max_iter MAX_ITER, -m MAX_ITER
                        Maximum number of iterations.
  --output OUTPUT, -o OUTPUT
                        Save layout to the specified file.
```

Example: 
```
python layout.py graphs/10square.dot -k 30 --output grid.pdf
```