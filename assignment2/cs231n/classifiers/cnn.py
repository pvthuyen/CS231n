import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = np.random.normal(0, weight_scale, (num_filters * H * W / 4, hidden_dim))
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    cache = [None, None, None, None]
    scores, cache[1] = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    scores, cache[2] = affine_relu_forward(scores, W2, b2)
    scores, cache[3] = affine_forward(scores, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    reg = self.reg
    loss, dx = softmax_loss(scores, y)
    loss += np.sum(0.5 * reg * W1**2)
    loss += np.sum(0.5 * reg * W2**2)
    loss += np.sum(0.5 * reg * W3**2)
    dx, grads['W3'], grads['b3'] = affine_backward(dx, cache[3]) 
    dx, grads['W2'], grads['b2'] = affine_relu_backward(dx, cache[2])
    dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx, cache[1])

    grads['W3'] += reg * W3
    grads['W2'] += reg * W2
    grads['W1'] += reg * W1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
pass

class ConvolutionalNet(object):
  def __init__(self, num_filters_list, filter_sizes, hidden_dims, 
               input_dim=(3, 32, 32), num_classes=10,
               use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32):
    self.use_batchnorm = use_batchnorm
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
      if use_batchnorm:
        self.params['gamma' + str(i + 1)] = np.random.rand(num_filters_list[i])
        self.params['beta' + str(i + 1)] = np.random.rand(num_filters_list[i])
      C = num_filters_list[i]
      H //= 2
      W //= 2
    self.params['W' + str(self.num_conv_layers + 1)] = np.random.normal(0, weight_scale, (C * H * W , hidden_dims[0]))
    self.params['b' + str(self.num_conv_layers + 1)] = np.zeros(hidden_dims[0])
    if use_batchnorm:
        self.params['gamma' + str(self.num_conv_layers + 1)] = np.random.rand(hidden_dims[0])
        self.params['beta' + str(self.num_conv_layers + 1)] = np.random.rand(hidden_dims[0])
    for i in range(1, self.num_fc_layers - 1):
      self.params['W' + str(self.num_conv_layers + 1 + i)] = np.random.normal(0, weight_scale, (hidden_dims[i - 1], hidden_dims[i]))
      self.params['b' + str(self.num_conv_layers + 1 + i)] = np.zeros(hidden_dims[i])
      if use_batchnorm:
        self.params['gamma' + str(self.num_conv_layers + 1 + i)] = np.random.rand(hidden_dims[i])
        self.params['beta' + str(self.num_conv_layers + 1 + i)] = np.random.rand(hidden_dims[i])
    self.params['W' + str(self.num_conv_layers + self.num_fc_layers)] = np.random.normal(0, weight_scale, (hidden_dims[-1], num_classes))
    self.params['b' + str(self.num_conv_layers + self.num_fc_layers)] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_conv_layers + self.num_fc_layers)]
    
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

    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

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
      # print i
      # print prev_out.shape, self.params['W' + str(i + 1)].shape
      filter_size = self.params['W' + str(i + 1)].shape[2]
      conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
      prev_out, conv_relu_pool_cache = conv_relu_pool_forward(prev_out, \
          self.params['W' + str(i + 1)], self.params['b' + str(i + 1)], conv_param, self.pool_param)
      spatial_batchnorm_cache = None
      if self.use_batchnorm:
        prev_out, spatial_batchnorm_cache = spatial_batchnorm_forward(prev_out, \
          self.params['gamma' + str(i + 1)], self.params['beta' + str(i + 1)], self.bn_params[i])
      cache.append((conv_relu_pool_cache, spatial_batchnorm_cache))

    for i in range(self.num_fc_layers):
      prev_out, affine_relu_cache = affine_relu_forward(prev_out,\
        self.params['W' + str(self.num_conv_layers + 1 + i)], self.params['b' + str(self.num_conv_layers + 1 + i)])
      # print i, prev_out.shape
      spatial_batchnorm_cache = None
      if i != self.num_fc_layers - 1 and self.use_batchnorm:
        prev_out, spatial_batchnorm_cache = batchnorm_forward(prev_out,\
          self.params['gamma' + str(self.num_conv_layers + 1 + i)], self.params['beta' + str(self.num_conv_layers + 1 + i)], self.bn_params[self.num_conv_layers + i])
      cache.append((affine_relu_cache, spatial_batchnorm_cache))
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
      if i != self.num_fc_layers - 1 and self.use_batchnorm:
        prev_dout, grads['gamma' + str(self.num_conv_layers + 1 + i)], grads['beta' + str(self.num_conv_layers + 1 + i)] =\
          batchnorm_backward(prev_dout, cache[-1][1])
        # print 'gamma' + str(self.num_conv_layers + 1 + i), grads['gamma' + str(self.num_conv_layers + 1 + i)].shape, self.params['gamma' + str(self.num_conv_layers + 1 + i)].shape
      prev_dout, grads['W' + str(self.num_conv_layers + 1 + i)], grads['b' + str(self.num_conv_layers + 1 + i)] =\
        affine_relu_backward(prev_dout, cache[-1][0])
      grads['W' + str(self.num_conv_layers + 1 + i)] += self.reg * W
      # print self.reg * W
      # if np.sum(grads['W' + str(self.num_conv_layers + 1 + i)]) == 0:
      #   print grads['W' + str(self.num_conv_layers + 1 + i)]
      cache.pop()

    for i in range(self.num_conv_layers - 1, -1, -1):
      W = self.params['W' + str(i + 1)]
      b = self.params['b' + str(i + 1)]
      loss += 0.5 * self.reg * np.sum(W ** 2)
      if self.use_batchnorm:
        prev_dout, grads['gamma' + str(i + 1)], grads['beta' + str(i + 1)] =\
          spatial_batchnorm_backward(prev_dout, cache[-1][1])
      prev_dout, grads['W' + str(i + 1)], grads['b' + str(i + 1)] = conv_relu_pool_backward(prev_dout, cache[-1][0])
      grads['W' + str(i + 1)] += self.reg * W
      # if np.sum(grads['W' + str(i + 1)]) == 0:
      #   print grads['W' + str(i + 1)]
      cache.pop()
    # print loss

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
    
pass