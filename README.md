#SUSTAIN-MATLAB

This set of scripts runs a minimal version of the SUSATIN model of category learning (Love, Medin, & Gureckis, 2004) for application in  traditional artificial classification learning (TACL) experiments. It is written in MATLAB, and is currently set up to simulate the six-types problem of Shepard, Hovland, & Jenkins (1961), though it should generalize to any dataset.



#####Keeping things as simple as possible:
This version of SUSTAIN is set up exclusively for class prediction within continuous (or at least binary) spaces. Feature prediction capabilities are not included, and neither is support for nominal dimensions. Todd Gureckis has a [nice implementation](https://github.com/NYUCCL/sustain_python) in python if you are looking for these capabilities.

There are a variety of utility scripts, and a few important ones:
- **start.m** can be used to set up and run SUSTAIN.
- **SUSTAIN.m** uses a provided architecture to train a network on a set of inputs and category assignments.
- **FORWARDPASS.m** and **UPDATE.m** are used to propagate activations forward through the model, and update the network's knowledge, respectively.

Simulations are run by executing the *start.m* script. All simulations begin by passing a model struct to the *SUSTAIN.m* function. At a minimum, 'model' needs to include:


| Field          | Description                               | Type                            |
| ---------------| ------------------------------------------| :-----------------------------: |
| `exemplars`    | Matrix of training stimuli coordinates    | Item-by-feature matrix          |
| `numblocks`    | Number of iterations over the exemplars   | Integer (>0)                    |
| `numinitials`  | Number of random initializations          | Integer (>0)                    |
| `assignments`  | class assignment for each exemplar        | Integer vector (>0)             |
| `params`       | [attn, comp, decision, lrate]             | Float vector (0 - Inf)          |

For almost all situations, inputs should be scaled to [-1 +1]. SUSTAIN.m will train the network and return a result struct. As-is, 'result' contains only training accuracy (averaged across inits) at each training block. Additional measures, such as test phase classification, can be added.

Written by [Nolan Conaway](http://bingweb.binghamton.edu/~nconawa1/).
Updated April 30, 2016

