import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *

class ConvolutionalNet(object):
  def __init__(self, num_filters_list, filter_sizes, hidden_dims, 
               input_dim=(3, 32, 32), num_classes=10,
               use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32):
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_conv_layers = len(num_filters_list)
    self.num_fc_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}
    self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    self.conv_param = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    C, H, W = input_dim
    for i in range(self.num_conv_layers):
      self.params['W' + str(i + 1)] = np.random.normal(0, weight_scale, (num_filters_list[i], C, filter_sizes[i], filter_sizes[i]))
      self.params['b' + str(i + 1)] = np.zeros(num_filters_list[i])
      C = num_filters_list[i]
      H //= 2
      W //= 2
    self.params['W' + str(self.num_conv_layers + 1)] = np.random.normal(0, weight_scale, (C * H * W , hidden_dims[0]))
    self.params['b' + str(self.num_conv_layers + 1)] = np.zeros(hidden_dims[0])
    for i in range(1, self.num_fc_layers - 1):
      self.params['W' + str(self.num_conv_layers + 1 + i)] = np.random.normal(0, weight_scale, (hidden_dims[i - 1], hidden_dims[i]))
      self.params['b' + str(self.num_conv_layers + 1 + i)] = np.zeros(hidden_dims[i])
    self.params['W' + str(self.num_conv_layers + self.num_fc_layers + 1)] = np.random.normal(0, weight_scale, (hidden_dims[-1], num_classes))
    self.params['b' + str(self.num_conv_layers + self.num_fc_layers + 1)] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
  
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)

  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    prev_out = X
    cache = []
    for i in range(self.num_conv_layers):
      filter_size = self.params['W' + str(i + 1)].shape[2]
      conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
      prev_out, conv_relu_pool_cache = conv_relu_pool_forward(X, self.params['W' + str(i + 1)], self.params['b' + str(i + 1)], conv_param, self.pool_param)
      cache.append(conv_relu_pool_cache)
    for i in range(self.num_fc_layers):
      prev_out, affine_relu_cache = affine_relu_forward(prev_out,\
        self.params['W' + str(self.num_conv_layers + 1 + i)], self.params['b' + str(self.num_conv_layers + 1 + i)])
      cache.append(affine_relu_cache)
    scores = prev_out

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, prev_dout = softmax_loss(scores, y)
    for i in range(self.num_fc_layers - 1, -1, -1):
      W = self.params['W' + str(self.num_conv_layers + 1 + i)]
      b = self.params['b' + str(self.num_conv_layers + 1 + i)]
      loss += 0.5 * self.reg * np.sum(W ** 2)
      prev_dout, grads['W' + str(self.num_conv_layers + 1 + i)], grads['b' + str(self.num_conv_layers + 1 + i)] =\
        affine_relu_backward(prev_dout, cache[-1])
      cache.pop()
    for i in range(self.num_conv_layers - 1, -1, -1):
      W = self.params['W' + str(i + 1)]
      b = self.params['b' + str(i + 1)]
      prev_dout, grads['W' + str(i + 1)], grads['b' + str(i + 1)] = conv_relu_pool_backward(prev_dout, cache[1])

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
    
pass