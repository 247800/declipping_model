# Restoration of the saturated signal using self-supervised learning
Welcome to the repositary of my diploma thesis project, which deals with self-supervised training of a bias-free U-net model for audio declipping.

The architecture follows https://arxiv.org/pdf/1505.04597 except the model consists of 5 convolution blocks in the encoder part and 5 up-convolution blocks in the decoder, while bias is not used.

Hyperparameters for training were set according to https://arxiv.org/pdf/2602.22279 section 5.3.

The training was conducted on the GTZAN dataset.
