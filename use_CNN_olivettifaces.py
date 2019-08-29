# -*-coding:utf8-*-#
"""
The functions implemented by this program:
In train_CNN_olivettifaces.py we trained and saved the parameters of the model, using these saved parameters to initialize the CNN model,
In this way, a usable CNN system is obtained, and a face map is input into the CNN system to predict the type of the face map.

@author:wepon(http://2hwp.com)
Explain the article of this code: http://blog.csdn.net/u012162613/article/details/43277187
"""

import os
import sys
import cPickle

import numpy
from PIL import Image

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv


#Read the previously saved training parameters
#layer0_params~layer3_params are all containing W and b, layer*_params[0] is W, layer*_params[1] is b
def load_params(params_file):
    f=open(params_file,'rb')
    layer0_params=cPickle.load(f)
    layer1_params=cPickle.load(f)
    layer2_params=cPickle.load(f)
    layer3_params=cPickle.load(f)
    f.close()
    return layer0_params,layer1_params,layer2_params,layer3_params

#read image, return face data of type numpy.array and corresponding label
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
    
    return faces,label



"""
LeNetConvPoolLayer, HiddenLayer, LogisticRegression in train_CNN_olivettifaces are randomly initialized
They are defined below as versions that can be initialized with parameters.
"""
class LogisticRegression(object):
    def __init__(self, input, params_W,params_b,n_in, n_out):
        self.W = params_W
        self.b = params_b
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


class HiddenLayer(object):
    def __init__(self, input, params_W,params_b, n_in, n_out,
                 activation=T.tanh):
        self.input = input
        self.W = params_W
        self.b = params_b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]

	
#convolution + sampling layer (conv+maxpooling)
class LeNetConvPoolLayer(object):
    def __init__(self,  input,params_W,params_b, filter_shape, image_shape, poolsize=(2, 2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.W = params_W
        self.b = params_b
        # 卷积
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )
        # 子采样
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]


"""
Initialize CNN with the previously saved parameters, you get a trained CNN model, and then use this model to measure the image.
Note: n_kerns is consistent with the previously trained model. The dataset is the path to the image you want to test, and params_file is the path to the parameter file saved before training.
"""
def use_CNN(dataset='olivettifaces.gif',params_file='params.pkl',nkerns=[5, 10]):   
    
   #Read the test image, here read the entire olivettifaces.gif, that is, all samples, get faces, label
    faces,label=load_data(dataset)
    face_num = faces.shape[0]   #How many faces are there?
  
    #Parameter
    layer0_params,layer1_params,layer2_params,layer3_params=load_params(params_file)
    
    x = T.matrix('x')  #Use the variable x to represent the input face data as input to layer0

    ######################
    #Initialize the parameters of each layer W and b with the parameters read in.
    ######################
    layer0_input = x.reshape((face_num, 1, 57, 47)) 
    layer0 = LeNetConvPoolLayer(
        input=layer0_input,
        params_W=layer0_params[0],
        params_b=layer0_params[1],
        image_shape=(face_num, 1, 57, 47),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    layer1 = LeNetConvPoolLayer(
        input=layer0.output,
        params_W=layer1_params[0],
        params_b=layer1_params[1],
        image_shape=(face_num, nkerns[0], 26, 21),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    layer2_input = layer1.output.flatten(2)
    layer2 = HiddenLayer(
        input=layer2_input,
        params_W=layer2_params[0],
        params_b=layer2_params[1],
        n_in=nkerns[1] * 11 * 8,
        n_out=2000,      
        activation=T.tanh
    )

    layer3 = LogisticRegression(input=layer2.output, params_W=layer3_params[0],params_b=layer3_params[1],n_in=2000, n_out=40)   
     
    #Define theano.function, let x be the input, layer3.y_pred (the predicted category) as the output
    f = theano.function(
        [x],    #funtion The input must be a list, even if there is only one input
        layer3.y_pred
    )
    
    #Predicted category pred
    pred = f(faces)
    

    #Compare the predicted category pred with the real category label and output a misclassified image
    for i in range(face_num): 
	 if pred[i] != label[i]:
                print('picture: %i is person %i, but mis-predicted as person %i' %(i, label[i], pred[i]))


if __name__ == '__main__':
	use_CNN()



"""A little note, the understanding of theano.function is not necessarily correct, and I will look back later.

In theano, you must define the input x and output through the function, and then call the function, and then start the calculation. For example, in use_CNN, when layer0 is defined, even if faces are defined as input, layer1~layer3 can be used directly. Layer3.y_pred to get the category.
Because in theano, layer0~layer3 is just a "graph" relationship, we define layer0~layer3, and only create this kind of graph relationship, but if there is no funtion, it will not be calculated.

This is why you want to define x:
     x = T.matrix('x')

Then use the variable x as the input to layer0.
Finally, define a function:
f = theano.function(
         [x], #funtion input must be list, even if there is only one input
         Layer3.y_pred
     )

Take x as input and layer3.y_pred as output.
When f(faces) is called, the predicted value is obtained.

"""
