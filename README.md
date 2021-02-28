## Factor analysis from incomplete data using EM

This work attempts to implement the EM algorithm specified in Roberts 2014 (https://doi.org/10.1016/j.csda.2013.08.018).
We structure our implementation similar to the approach in Ghahramani and Hinton (http://www.cs.utoronto.ca/~hinton/absps/tr-96-1.pdf).

### Installation

1. Change to the src directory: ```cd src```
2. Install the necessary libraries by running ```pip requirements.txt```

The program can be run (using the student test scores example given in Table 1 of Roberts 2014) by running ```python ffa_roberts.py```
It will print the log-likelihood for each iteration, followed by the parameters.