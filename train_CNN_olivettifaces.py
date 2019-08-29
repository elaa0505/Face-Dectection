# -*-coding:utf8-*-#
"""
This program is based on python+numpy+theano+PIL development. It uses a CNN model similar to LeNet5 and is applied to the olivettifaces face database.
The function of face recognition is realized, and the error of the model is reduced to less than 5%.
This program is just a toy implement for the personal learning process. The model may have overfitting, because the sample is small, and this is not verified.
However, the program is intended to clarify the specific steps of the program to develop the CNN model, especially for image recognition, from getting the image database to implementing a CNN model for this image database.
I think this program has a reference to the implementation of these processes.
@author:wepon(http://2hwp.com)
An article explaining this code：http://blog.csdn.net/u012162613/article/details/43277187
"""
import os
import sys
import time

import numpy
from PIL import Image

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

"""
The function that loads the image data, dataset_path is the path of the image olivettifaces
After loading olivettifaces, it is divided into three data sets: train_data, valid_data, and test_data.
The function returns train_data, valid_data, test_data and the corresponding label
"""
def load_data(dataset_path):
    img = Image.open(dataset_path)
    img_ndarray = numpy.asarray(img, dtype='float64')/256
    faces=numpy.empty((400,2679))
    for row in range(20):
	   for column in range(20):
		faces[row*20+column]=numpy.ndarray.flatten(img_ndarray [row*57:(row+1)*57,column*47:(column+1)*47])

    label=numpy.empty(400)
    for i in range(40):
	label[i*10:i*10+10]=i
    label=label.astype(numpy.int)

    #Divided into training set, verification set, test set, the size is as follows
    train_data=numpy.empty((320,2679))
    train_label=numpy.empty(320)
    valid_data=numpy.empty((40,2679))
    valid_label=numpy.empty(40)
    test_data=numpy.empty((40,2679))
    test_label=numpy.empty(40)

    for i in range(40):
	train_data[i*8:i*8+8]=faces[i*10:i*10+8]
	train_label[i*8:i*8+8]=label[i*10:i*10+8]
	valid_data[i]=faces[i*10+8]
	valid_label[i]=label[i*10+8]
	test_data[i]=faces[i*10+9]
	test_label[i]=label[i*10+9]

    #Defining a dataset as a shared type allows data to be copied into the GPU, leveraging the GPU to speed up the program.
    def shared_dataset(data_x, data_y, borrow=True):
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')



    train_set_x, train_set_y = shared_dataset(train_data,train_label)
    valid_set_x, valid_set_y = shared_dataset(valid_data,valid_label)
    test_set_x, test_set_y = shared_dataset(test_data,test_label)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval



#The classifier, the last layer of CNN, uses logistic regression (softmax)
class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


#Fully connected layer, layer before the classifier
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):

        self.input = input

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


#Convolution + sampling layer（conv+maxpooling）
class LeNetConvPoolLayer(object):

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):

        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))

        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolution
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # Subsampling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


#Function to save training parameters
def save_params(param1,param2,param3,param4):  
        import cPickle  
        write_file = open('params.pkl', 'wb')   
        cPickle.dump(param1, write_file, -1)
        cPickle.dump(param2, write_file, -1)
        cPickle.dump(param3, write_file, -1)
        cPickle.dump(param4, write_file, -1)
        write_file.close()  



"""
The basic components of CNN are defined above. The following function applies CNN to the data set of olivettifaces. The model of CNN is based on LeNet.
The optimization algorithm used is the batch random gradient descent algorithm, minibatch SGD, so many of the following parameters have batch_size, such as image_shape=(batch_size, 1, 57, 47)
The parameters that can be set are:
Batch_size, but note that the calculations for n_train_batches, n_valid_batches, and n_test_batches all depend on batch_size
Nkerns=[5, 10], that is, the number of convolution kernels of the first two layers can be set
The number of output neurons n_out of the fully connected layer HiddenLayer can be set, and the input of the classifier must be changed at the same time.
In addition, another important thing is the learning rate learning_rate.
"""

def evaluate_olivettifaces(learning_rate=0.05, n_epochs=200,
                    dataset='olivettifaces.gif',
                    nkerns=[5, 10], batch_size=40):   

    #Random number generator for initializing parameters
    rng = numpy.random.RandomState(23455)
    #Download Data
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    #Calculate the number of batches for each data set
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    #Define several variables, x for face data, as input for layer0
    index = T.lscalar()
    x = T.matrix('x')  
    y = T.ivector('y')



    ######################
    #Establish a CNN model:
    #input+layer0(LeNetConvPoolLayer)+layer1(LeNetConvPoolLayer)+layer2(HiddenLayer)+layer3(LogisticRegression)
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 57 * 47)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (57, 47) is the size of  images.
    layer0_input = x.reshape((batch_size, 1, 57, 47))

    # First convolution + maxpool layer
    # Obtained after convolution：(57-5+1 , 47-5+1) = (53, 43)
    # maxpooling After getting： (53/2, 43/2) = (26, 21)，Because the boundary is ignored
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 26, 21)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 57, 47),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    # The second convolution + maxpool layer, the input is the output of the upper layer, ie (batch_size, nkerns[0], 26, 21)
    # After convolution, get: (26-5+1, 21-5+1) = (22, 17)
    # After maxpooling you get: (22/2, 17/2) = (11, 8) because the boundary is ignored
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 11, 8)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 26, 21),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    # HiddenLayer fully connected layer, its input size is (batch_size, num_pixels), which means that each sample is formed into a one-dimensional long vector after layer0 and layer1.，   
    #There are batch_size samples, so the input size is (batch_size, num_pixels), and each line is a long vector of samples.
    #So convert the output of the previous layer (batch_size, nkerns[1], 11, 8) to (batch_size, nkerns[1] * 11* 8), using flatten
    layer2_input = layer1.output.flatten(2)
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 11 * 8,
        n_out=2000,      #The number of neurons in the fully connected layer is defined by itself and can be adjusted as needed.
        activation=T.tanh
    )

    #Classifier
    layer3 = LogisticRegression(input=layer2.output, n_in=2000, n_out=40)   #N_in is equal to the output of the fully connected layer, n_out is equal to 40 categories


    ###############
    # Define some basic elements of the optimization algorithm: cost function, training, verification, test model, parameter update rule (ie gradient drop)
    ###############
    # Cost function
    cost = layer3.negative_log_likelihood(y)
    
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # All parameters
    params = layer3.params + layer2.params + layer1.params + layer0.params
    #Gradient of each parameter
    grads = T.grad(cost, params)
    #Parameter update rule
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
    #train_model Optimize update parameters according to MSGD during training
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )


    ###############
    # Train the CNN stage to find the optimal parameters.
    ###############
    print '... training'
    #In LeNet5, batch_size=500, n_train_batches=50000/500=100, patience=10000
    #In olivettifaces, batch_size=40, n_train_batches=320/40=8, paticence can be set to 800 accordingly, this can be adjusted according to the actual situation, it does not matter if it is adjusted up.
    patience = 800
    patience_increase = 2  
    improvement_threshold = 0.99  
    validation_frequency = min(n_train_batches, patience / 2) 


    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)
            
            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
		    save_params(layer0.params,layer1.params,layer2.params,layer3.params)#Save parameter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))




if __name__ == '__main__':
	evaluate_olivettifaces()
