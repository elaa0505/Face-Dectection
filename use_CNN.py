# -*-coding:utf8-*-#
"""
The functions implemented by this program:
In train_CNN_olivettifaces.py we trained and saved the parameters of the model, using these saved parameters to initialize the CNN model,
In this way, a usable CNN system is obtained, and a face map is input into the CNN system to predict the type of the face map.

@author:wepon(http://2hwp.com)
Explain the article of this code: http://blog.csdn.net/u012162613/article/details/43277187
"""

Import os
Import sys
Import cPickle

Import numpy
From PIL import Image

Import theano
Import theano.tensor as T
From theano.tensor.signal import downsample
From theano.tensor.nnet import conv


#Read the previously saved training parameters
#layer0_params~layer3_params are all containing W and b, layer*_params[0] is W, layer*_params[1] is b
Def load_params(params_file):
    f=open(params_file,'rb')
    Layer0_params=cPickle.load(f)
    Layer1_params=cPickle.load(f)
    Layer2_params=cPickle.load(f)
    Layer3_params=cPickle.load(f)
    F.close()
    Return layer0_params, layer1_params, layer2_params, layer3_params

#read image, return face data of type numpy.array and corresponding label
Def load_data(dataset_path):
    Img = Image.open(dataset_path)
    Img_ndarray = numpy.asarray(img, dtype='float64')/256

    Faces=numpy.empty((400,2679))
    For row in range(20):
For column in range(20):
Faces[row*20+column]=numpy.ndarray.flatten(img_ndarray [row*57:(row+1)*57,column*47:(column+1)*47])

    Label=numpy.empty(400)
    For i in range(40):
Label[i*10:i*10+10]=i
    Label=label.astype(numpy.int)
    
    Return faces,label



"""
LeNetConvPoolLayer, HiddenLayer, LogisticRegression in train_CNN_olivettifaces are randomly initialized
They are defined below as versions that can be initialized with parameters.
"""
Class LogisticRegression(object):
    Def __init__(self, input, params_W, params_b, n_in, n_out):
        self.W = params_W
        Self.b = params_b
        Self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        Self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        Self.params = [self.W, self.b]

    Def negative_log_likelihood(self, y):
        Return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    Def errors(self, y):
        If y.ndim != self.y_pred.ndim:
            Raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        If y.dtype.startswith('int'):
            Return T.mean(T.neq(self.y_pred, y))
        Else:
            Raise NotImplementedError()


Class HiddenLayer(object):
    Def __init__(self, input, params_W, params_b, n_in, n_out,
                 Activation=T.tanh):
        Self.input = input
        self.W = params_W
        Self.b = params_b

        Lin_output = T.dot(input, self.W) + self.b
        Self.output = (
            Lin_output if activation is None
            Else activation(lin_output)
        )
        Self.params = [self.W, self.b]


#convolution + sampling layer (conv+maxpooling)
Class LeNetConvPoolLayer(object):
    Def __init__(self, input, params_W, params_b, filter_shape, image_shape, poolsize=(2, 2)):
        Assert image_shape[1] == filter_shape[1]
        Self.input = input
        self.W = params_W
        Self.b = params_b
        # convolution
        Conv_out = conv.conv2d(
            Input=input,
            Filters=self.W,
            Filter_shape=filter_shape,
            Image_shape=image_shape
        )
        #子采样
        Pooled_out = downsample.max_pool_2d(
            Input=conv_out,
            Ds=poolsize,
            Ignore_border=True
        )
        Self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        Self.params = [self.W, self.b]


"""
Initialize CNN with the previously saved parameters, you get a trained CNN model, and then use this model to measure the image.
Note: n_kerns is consistent with the previously trained model. The dataset is the path to the image you want to test, and params_file is the path to the parameter file saved before training.
"""
Def use_CNN(dataset='olivettifaces.gif',params_file='params.pkl',nkerns=[5, 10]):
    
    #Read the test image, here read the entire olivettifaces.gif, that is, all samples, get faces, label
    Faces,label=load_data(dataset)
    Face_num = faces.shape[0] #How many faces are there?
  
    #读入Parameter
    Layer0_params, layer1_params, layer2_params, layer3_params=load_params(params_file)
    
    x = T.matrix('x') #Use the variable x to represent the input face data as input to layer0

    ######################
    # Initialize the parameters of each layer W and b with the parameters read in.
    ######################
    Layer0_input = x.reshape((face_num, 1, 57, 47))
    Layer0 = LeNetConvPoolLayer(
        Input=layer0_input,
        params_W=layer0_params[0],
        Params_b=layer0_params[1],
        Image_shape=(face_num, 1, 57, 47),
        Filter_shape=(nkerns[0], 1, 5, 5),
        Poolsize=(2, 2)
    )

    Layer1 = LeNetConvPoolLayer(
        Input=layer0.output,
        params_W=layer1_params[0],
        Params_b=layer1_params[1],
        Image_shape=(face_num, nkerns[0], 26, 21),
        Filter_shape=(nkerns[1], nkerns[0], 5, 5),
        Poolsize=(2, 2)
    )

    Layer2_input = layer1.output.flatten(2)
    Layer2 = HiddenLayer(
        Input=layer2_input,
        params_W=layer2_params[0],
        Params_b=layer2_params[1],
        N_in=nkerns[1] * 11 * 8,
        N_out=2000,
        Activation=T.tanh
    )

    Layer3 = LogisticRegression(input=layer2.output, params_W=layer3_params[0],params_b=layer3_params[1],n_in=2000, n_out=40)
     
    # Define theano.function, let x be the input, layer3.y_pred (the predicted category) as the output
    f = theano.function(
        [x]
