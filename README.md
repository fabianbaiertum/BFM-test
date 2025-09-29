# BFM-test

It implements the statistical test of Brueck, Fermanian and Min published in their paper Distribution free MMD tests for model selection with estimated parameters (https://arxiv.org/abs/2305.07549) in Python. 
Their paper was later accepted by the Journal of Machine Learning Research (https://www.jmlr.org/papers/volume26/23-1199/23-1199.pdf).

It's a test for model selection and specification, given i.i.d. samples X, Y and optionally Y2. 

I'm using the R code of https://github.com/florianbrueck/MMD_tests_for_model_selection/blob/main/BFM-TEST.R for most of my calculations. I made a vectorised version of it and also has a version for GPU usage. I didn't use their code for the figure generation. For data generation, I used the same logic as in their paper.

For the functions kern_k and MMD_stand, I used something similar to https://github.com/sshekhar17/PermFreeMMD/blob/main/src/utils.py#L76.

My code improved the performance of the provided R code using Numpy by ~60 times, using GPU, it improves by at least ~70 times compared to the Numpy code (overall improvement of over 4200 times). 
For the GPU version, you need to install the CUDA package from Nvidia and then install Cupy in your IDE.

I also included my paper for the research internship at the Financial Mathematics Department at Technical University of Munich (TUM).






#### Remarks:
For large n, the code needs to be adjusted so that you don't run into GPU memory issues. Also, an O(n) MMD estimator could be considered.
The provided CSV files x, y1 and y2 were used to test that the implementation of the BFM test is correct.
The test can easily be run with your own data input for X, Y1 and Y2; you have to change the path to where your CSV files lie.
