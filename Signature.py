import numpy as np
from PIL import Image, ImageOps
import os
import random
import csv
import math as m


'''
Helpful resources:
https://thesai.org/Downloads/Volume11No7/Paper_19-Handwriting_Recognition_using_Artificial_Intelligence.pdf
http://article.nadiapub.com/IJSIP/vol8_no4/14.pdf

This class just contains the neural network to analyze signatures. It will process images, map them to arrays, 
learn through backpropogation, and save progress to a file.

Learning Process Step by Step:
1. Take random image from train and its corresponding value from written_name_train
2. Crop the image and map the image into some 1d array
3. Run that array as input to the NN and record all outputs from all hidden nodes.
4. Use those results and the actual value of the name to back propogate

- The output array should be a binary string which can then be converted into characters (I think, not so sure).
- I'm pretty sure the input length needs to be predetermined, so I will define the largest photo in the data to be 
the minimum size, and then every image will be reverse cropped to fit that size.
- I don't know how many layers this will take but I will start with 3. The input layer will have however many nodes as
pixels in the image, and there will be two more layers after that. I don't know how many nodes I should have in them.
- I want this to be dynamically sized, so I can change it easily. 
'''
class Signature():

    dataLength = 0
    # 250 70
    INPUT_LENGTH = 17500
    dataRows = []
    network = []
    alpha = 0.5
    numberHiddenLayers = 3
    nodesPerLayer = [17500, 7000, 1000]
    dataPath = 'C:/Datasets/signatures/'
    csvPath = os.getcwd()+'/answerkey.csv'
    weightPath = os.getcwd()+'/weights'

    def __init__(self):
        # print(self.find_max_height_in_data())

        self.init_training_data()
        sample = self.grab_sample()
        a = self.clean_image(sample[0])

        pass

    # Handles all of the initializations
    def init_network(self):
        # There are INPUT_LENGTH inputs (17500 right now).

        for i in range(self.numberHiddenLayers):
            layer = np.array()


            pass

    # Takes the final output of the network and sees how close it is
    def error(self, array, answer):
        #

        pass

    # Given x, return f'(x) where f is the sigmoid function
    def sigmoid_derivative(self, out):
        return out * (1 - out)

    # Perform backpropogation on the NN
    def backpropogation(self, input_list, output_list, expected_list):

        # Calculate Errors
        output_error = self.error(output_list, expected_list)


        # Readjust Weights

        return

    # Initialize the dataset (Get number of images stored)
    def init_training_data(self):
        with open(self.csvPath, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                self.dataRows.append(row)
            self.dataLength = reader.line_num

    # Returns the image filename and correct spelling as a tuple (Filename, Identity)
    def grab_sample(self):
        a = random.randint(0, self.dataLength)
        return self.dataRows[a]

    # Maps images to 1-D arrays
    def map_image_to_array(self, image_file_name):
        im = Image.open(self.dataPath + image_file_name)
        w, h = im.size

        # TODO: When Cropping larger than needed, change all black pixels to white
        # TODO: Segment the signatures (find where letters start and end)
        # This might be too much for this project so I'll just add another layer or something who knows

        # This could cause problems later but i hope it doesn't
        # Also, if you are cropping an image to a larger size, the extra pixels are black. Maybe these should be white.
        cropped = ImageOps.invert(im.convert("1")).crop((0, 0, 250, 70))
        cropped.show()
        arr = np.array(cropped)
        return arr.ravel()  # This is a vector of real numbers

    # Using an input array, calculate the network's output. Should return the outputs of each layer
    # inputArray should be a numpy array
    def get_network_output(self, input_array):
        outputs = np.array([])
        np.append(outputs, input_array)
        for i, layer in enumerate(self.network):
            np.append(outputs, self.eval_layer(outputs[i], layer))
            pass

    # Given a layer of nodes and input to those nodes, return the output array
    def eval_layer(self, input_array, layer):
        return self.activation()

    def write_weights_to_file(self):
        pass

    def read_weights_from_file(self):
        pass

    # input should be a numpy array
    def activation(self, i):
        return 1/(1+m.exp(-i))


    # (72, 388) most of the width is whitespace. I think the best way to do this is to lower the resolution and/or
    # crop the right side of the photo within reasonable range
    def find_max_height_in_data(self):
        images = os.listdir(self.dataPath)
        images.sort()
        maxHeight = 0
        maxWidth = 0
        for file in images:
            print(file)
            im = np.array(Image.open(self.dataPath+file))
            maxHeight = max(maxHeight, im.shape[0])
            maxWidth = max(maxWidth, im.shape[1])
        return maxHeight, maxWidth
