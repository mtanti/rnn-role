# rnn-role
Code used by the paper "What is the Role of Recurrent Neural Networks (RNNs) in an Image Caption Generator?".

This paper investigates whether the RNN should be viewed as a sequence generator or as a sequence encoder. It does so by comparing the performance of a caption generator when the image features are included in the RNN or outside of the RNN. Paper will appear at the INLG 2017 conference.

Works on both python 2 and python 3 (except for the MSCOCO evaluation toolkit which requires python 2).

Python dependencies (install all with pip):

    tensorflow
    future
