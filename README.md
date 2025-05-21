# BFM-test

It implements the statistical test of Brueck, Fermanian and Min published in their paper Distribution free MMD tests for model selection with estimated parameters (https://arxiv.org/abs/2305.07549) in Python. 
I'm using the R code of https://github.com/florianbrueck/MMD_tests_for_model_selection/blob/main/BFM-TEST.R for most of my calculations. I vectorized it and also have a version for GPU usage. I didn't use their code for the figure generation. For data generation, I used the same logic as in their paper.

For the functions kern_k and MMD_stand, I used something similar to https://github.com/sshekhar17/PermFreeMMD/blob/main/src/utils.py#L76.

My code improved the performance of the provided R code using Numpy by ~60 times, using GPU, it improves by at least ~70 times compared to the Numpy code.

The test can easily be run with your own data input for X, Y1 and Y2; you have to change the path to where your csv files lie.



#### Remark
For large n the code needs to be adjusted, so that you don't run into GPU memory issues

The provided csv files x, y1 and y2 were used to test that the implementation of the BFM test is correct.
