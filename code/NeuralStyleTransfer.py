import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image

import numpy as np
import tensorflow as tf

  


class pittore:
    def __init__(self):
        self.__IMAGE_WIDTH    = 400 # to make input width match the VGG-model input width
        self.__IMAGE_HEIGHT   = 300 # to make input height match the VGG-model input height
        self.__COLOR_CHANNELS = 3 # RBG
        self.__NOISE_RATIO    = 0.6 # parameter to use based on "Very Deep Convolutional Networks for Large-Scale Image Recognition". 
        self.__MEANS          = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) # parameter to use based on "Very Deep Convolutional Networks for Large-Scale Image Recognition". 
        self.__VGG_MODEL      = 'pretrained-model/imagenet-vgg-verydeep-19.mat' # Pick the VGG 19-layer model by from the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition".
        self.__OUTPUT_DIR     = 'output/' #output directory
        self.__vgg            = scipy.io.loadmat(self.__VGG_MODEL) # Loading the model
        self.__vgg_layers     = self.__vgg['layers'] # layer names
        #self.__STYLE_LAYERS   = [('conv1_1', 0.2),('conv2_1', 0.2),('conv3_1', 0.2),('conv4_1', 0.2),('conv5_1', 0.2)]
        
        
        self.STYLE_IMAGE    = 'images/stone_style.jpg' # Style image to use.
        self.CONTENT_IMAGE  = 'images/content300.jpg' # Content image to use.

    def __weights(self,layer, expected_layer_name):
        """
        Return the weights and bias from the VGG model for a given layer.
        """
        wb = self.__vgg_layers[0][layer][0][0][2]
        W  = wb[0][0]
        b  = wb[0][1]
        layer_name = self.__vgg_layers[0][layer][0][0][0][0]
        assert layer_name == expected_layer_name
        return W, b

            

    def __relu(self,conv2d_layer):
        """
        Return the RELU function wrapped over a TensorFlow layer. Expects a
        Conv2d layer input.
        """
        return tf.nn.relu(conv2d_layer)

    def __conv2d(self,prev_layer, layer, layer_name):
        """
        Return the Conv2D layer using the weights, biases from the VGG
        model at 'layer'.
        """
        W, b = self.__weights(layer, layer_name)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        return tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

    def __conv2d_relu(self,prev_layer, layer, layer_name):
        """
        Return the Conv2D + RELU layer using the weights, biases from the VGG
        model at 'layer'.
        """
        return self.__relu(self.__conv2d(prev_layer, layer, layer_name))

    def __avgpool(self,prev_layer):
        """
        Return the AveragePooling layer.
        """
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    def __load_vgg_model(self,path):
        """
        Returns a model for the purpose of 'painting' the picture.
        Takes only the convolution layer weights and wrap using the TensorFlow
        Conv2d, Relu and AveragePooling layer. VGG actually uses maxpool but
        the paper indicates that using AveragePooling yields better results.
        The last few fully connected layers are not used.
        Here is the detailed configuration of the VGG model:
            0 is conv1_1 (3, 3, 3, 64)
            1 is relu
            2 is conv1_2 (3, 3, 64, 64)
            3 is relu    
            4 is maxpool
            5 is conv2_1 (3, 3, 64, 128)
            6 is relu
            7 is conv2_2 (3, 3, 128, 128)
            8 is relu
            9 is maxpool
            10 is conv3_1 (3, 3, 128, 256)
            11 is relu
            12 is conv3_2 (3, 3, 256, 256)
            13 is relu
            14 is conv3_3 (3, 3, 256, 256)
            15 is relu
            16 is conv3_4 (3, 3, 256, 256)
            17 is relu
            18 is maxpool
            19 is conv4_1 (3, 3, 256, 512)
            20 is relu
            21 is conv4_2 (3, 3, 512, 512)
            22 is relu
            23 is conv4_3 (3, 3, 512, 512)
            24 is relu
            25 is conv4_4 (3, 3, 512, 512)
            26 is relu
            27 is maxpool
            28 is conv5_1 (3, 3, 512, 512)
            29 is relu
            30 is conv5_2 (3, 3, 512, 512)
            31 is relu
            32 is conv5_3 (3, 3, 512, 512)
            33 is relu
            34 is conv5_4 (3, 3, 512, 512)
            35 is relu
            36 is maxpool
            37 is fullyconnected (7, 7, 512, 4096)
            38 is relu
            39 is fullyconnected (1, 1, 4096, 4096)
            40 is relu
            41 is fullyconnected (1, 1, 4096, 1000)
            42 is softmax
        """

        # Constructs the graph model.
        graph = {}
        graph['input']   = tf.Variable(np.zeros((1, self.__IMAGE_HEIGHT, self.__IMAGE_WIDTH, self.__COLOR_CHANNELS)), dtype = 'float32')
        graph['conv1_1']  = self.__conv2d_relu(graph['input'], 0, 'conv1_1')
        graph['conv1_2']  = self.__conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
        graph['avgpool1'] = self.__avgpool(graph['conv1_2'])
        graph['conv2_1']  = self.__conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
        graph['conv2_2']  = self.__conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
        graph['avgpool2'] = self.__avgpool(graph['conv2_2'])
        graph['conv3_1']  = self.__conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
        graph['conv3_2']  = self.__conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
        graph['conv3_3']  = self.__conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
        graph['conv3_4']  = self.__conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
        graph['avgpool3'] = self.__avgpool(graph['conv3_4'])
        graph['conv4_1']  = self.__conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
        graph['conv4_2']  = self.__conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
        graph['conv4_3']  = self.__conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
        graph['conv4_4']  = self.__conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
        graph['avgpool4'] = self.__avgpool(graph['conv4_4'])
        graph['conv5_1']  = self.__conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
        graph['conv5_2']  = self.__conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
        graph['conv5_3']  = self.__conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
        graph['conv5_4']  = self.__conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
        graph['avgpool5'] = self.__avgpool(graph['conv5_4'])

        return graph

    def __compute_content_cost(self,a_C, a_G):
        """
        Computes the content cost between the Image_C and Image_G
        
        Arguments:
            a_C <- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
            a_G <- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
        
        Returns: 
            J_content <- scalar that you compute using equation 1 above.
        """
        # Retrieve dimensions from a_G (get_shape() is tensor function)
        m, n_H, n_W, n_C =  a_G.get_shape().as_list() # This is a tensor function
        
        # Reshape a_C and a_G 
        a_C_unrolled = tf.reshape(a_C,(n_H*n_W,n_C))
        a_G_unrolled = tf.reshape(a_G,(n_H*n_W,n_C))
        
        # compute the cost with tensorflow (≈1 line)
        J_content = 1/(4*n_W*n_C*n_H)*tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled,a_G_unrolled)))
        
        return J_content

    

    def __gram_matrix(self,A):
        """
        This function is also called the unormalized covariance - used to calculate style dissimilarity 

        Argument:
        A -- matrix of shape (n_C, n_H*n_W)
        
        Returns:
        GA -- Gram matrix of A, of shape (n_C, n_C)
        """
        
        GA = tf.matmul(A,tf.transpose(A))
        return GA

    def __compute_layer_style_cost(self,a_S, a_G):
        """
        Calculates the style dissimilarity between Image_S and Image_G for one layer 
        
        Arguments:
        a_S <- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
        a_G <- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
        
        Returns: 
        J_style_layer <- tensor representing a scalar value, style cost defined above by equation (2)
        """
        # Retrieve dimensions from a_G 
        m, n_H, n_W, n_C = a_G.get_shape().as_list() # This is tensor function
        
        # Reshape the images to have them of shape (n_C, n_H*n_W) 
        a_S = tf.transpose(tf.reshape(a_S,(n_H*n_W,n_C)))
        a_G = tf.transpose(tf.reshape(a_G,(n_H*n_W,n_C)))

        # Computing gram_matrices for both images S and G
        GS = self.__gram_matrix(a_S)
        GG = self.__gram_matrix(a_G)
        
        # Computing the loss 
        J_style_layer = tf.square((1/(2*n_C*n_W*n_H))) * tf.reduce_sum(tf.square(tf.subtract(GG,GS)))
        
        return J_style_layer

    
    def __compute_style_cost(self,model,STYLE_LAYERS,sess):
        """
        Computes the overall style cost from several chosen layers
        
        Arguments:
        model <- our tensorflow model
        STYLE_LAYERS <- A python list containing:
                            - the names of the layers we would like to extract style from
                            - a coefficient for each of them
        
        Returns: 
        J_style -- tensor representing a scalar value, style cost defined above by equation (2)
        """
        
        
        J_style = 0

        for layer_name, coeff in STYLE_LAYERS:

            # Select the output tensor of the currently selected layer
            out = model[layer_name]

            # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
            a_S = sess.run(out)

            # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
            # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
            # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
            a_G = out
            
            # Compute style_cost for the current layer
            J_style_layer = self.__compute_layer_style_cost(a_S, a_G)

            # Add coeff * J_style_layer of this layer to overall style cost
            J_style += coeff * J_style_layer

        return J_style

    def __total_cost(self,J_content, J_style, alpha = 10, beta = 40):
        """
        Computes the total cost function
        
        Arguments:
        J_content -- content cost coded above
        J_style -- style cost coded above
        alpha -- hyperparameter weighting the importance of the content cost
        beta -- hyperparameter weighting the importance of the style cost
        
        Returns:
        J -- total cost as defined by the formula above.
        """
        
       
        J = alpha * J_content + beta * J_style
        
        return J

    def __reshape_and_normalize_image(self,image):
        """
        Reshape and normalize the input image (content or style)
        """
        
        # Reshape image to mach expected input of VGG19
        image = np.reshape(image, ((1,) + image.shape))
        
        # Substract the mean to match the expected input of VGG19
        image = image - self.__MEANS
        
        return image


    def __image_from_array(self,image):
        
        # Un-normalize the image so that it looks good
        image = image + self.__MEANS
        
        # Clip and Save the image
        image = np.clip(image[0], 0, 255).astype('uint8')
        img = Image.fromarray(image, 'RGB')
        return(img)

    def __generate_noise_image(self,content_image):
        """
        Generates a noisy image by adding random noise to the content_image
        """
        
        # Generate a random noise_image
        noise_image = np.random.uniform(-20, 20, (1, self.__IMAGE_HEIGHT, self.__IMAGE_WIDTH, self.__COLOR_CHANNELS)).astype('float32')
        
        # Set the input_image to be a weighted average of the content_image and a noise_image
        input_image = noise_image * self.__NOISE_RATIO + content_image * (1 - self.__NOISE_RATIO) 
    
        return input_image

    def creareArte(self,IMAGE_CONTENT,IMAGE_STYLE,content_layer='conv4_2',style_coef=[('conv1_1', 0.2),('conv2_1', 0.2),('conv3_1', 0.2),('conv4_1', 0.2),('conv5_1', 0.2)],epochs=300,verbose=True):
        """
        Generates new Images created based on the journal article titled: A Neural Algorithm of Artistic Style

        Arguments:
            IMAGE_CONTENT   <- JPG or JPEG format image representing the content image
            IMAGE_STYLE     <- JPG or JPEG format image representing the style image
        Return:
            IMAGE_GENERATED <- JPG or JEPG format image representing the generated image 

        """
        # resize the content and style images to match VGG19 model input
        im_c = np.array(IMAGE_CONTENT.resize((self.__IMAGE_WIDTH,self.__IMAGE_HEIGHT))) 
        im_s = np.array(IMAGE_STYLE.resize((self.__IMAGE_WIDTH,self.__IMAGE_HEIGHT)))

        # Reshape and normalize content image
        im_c = self.__reshape_and_normalize_image(im_c)

        # Reshape and normalize style image
        im_s = self.__reshape_and_normalize_image(im_s)

        # Generate input image correlated to content image
        im_g = self.__generate_noise_image(im_c)

        # reset the graph model
        tf.compat.v1.reset_default_graph()

        # start interactive session
        sess = tf.compat.v1.InteractiveSession()


        ########################################## Content Cost ###############################
        # loading the model
        model = self.__load_vgg_model(path = 'imagenet-vgg-verydeep-19.mat')

        # Assign the content image to be the input of the VGG model.  
        sess.run(model['input'].assign(im_c))

        # Select the output tensor of layer conv4_2
        out = model[content_layer]

        # Set a_C to be the hidden layer activation from the layer we have selected
        a_C = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out

        # Compute the content cost
        J_content = self.__compute_content_cost(a_C, a_G)

        ########################################## Style Cost ###############################

        # Assign the input of the model to be the "style" image 
        sess.run(model['input'].assign(im_s))

        # Compute the style cost
        J_style = self.__compute_style_cost(model, style_coef,sess=sess)

        ########################################## Total Cost ###############################
        J = self.__total_cost(J_content, J_style, alpha = 10, beta = 40)
        
        ########################################## Optimizer ###############################
        # define optimizer
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=2.0)
        
        # define train_step
        train_step = optimizer.minimize(J)

        ########################################## Model_nn ###############################
        #initialize global variable
        sess.run(tf.compat.v1.global_variables_initializer())

        # Run the noisy input image im_g
        sess.run(model["input"].assign(im_g))

        generated_images_list=[]
        # looping over the epochs for the optimization process
        for i in range(epochs):
            # run the session on the training step
            sess.run(fetches=train_step)

            # Compute the generated image by running the session on the current model['input']
            generated_image_array = sess.run(model['input'])
            generated_image = self.__image_from_array(generated_image_array)
            if i%10 == 0:
                Jt, Jc, Js = sess.run([J, J_content, J_style])
                if verbose==True:
                    print("Iteration " + str(i) + " :")
                    print("total cost = " + str(Jt))
                    print("content cost = " + str(Jc))
                    print("style cost = " + str(Js))
                generated_images_list.append(generated_image)
                
            
        
        sess.close()
        del sess

        
        return generated_images_list
        





