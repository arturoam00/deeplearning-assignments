### Part One of the CNNs DL assignment @ VU

The code for the assignment questions can be found under `src`. Here is the correspondence of questions and code: 
    
- Question 1 -> `conv.py`: Implements a convolution with `for` loops.

- Question 3 -> `unfold.py`: Implements the `unfold` function with for loops. 

- Question 4 -> `conv_layer_module.py`: This file contains `Conv2DModule`, which just implements the forward pass (backward is already done by `torch.nn.Module`) 

- Question 5, 6 -> `conv_layer_func.py`: This file contains `Conv2DFunc`, which inherits from `torch.nn.functional` and implements both the forward and the backward pass. 

From the directory where this file is, run:
```bash
pytest
```
To test all the code under `src` checking with the expected results from Pytorch. (You may need to install `pytest`)