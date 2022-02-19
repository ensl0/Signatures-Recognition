import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import os
import random
import csv
import math as m
from scipy import ndimage



'''
Helpful resources:
https://thesai.org/Downloads/Volume11No7/Paper_19-Handwriting_Recognition_using_Artificial_Intelligence.pdf
http://article.nadiapub.com/IJSIP/vol8_no4/14.pdf
http://dangminhthang.com/computer-vision/characters-segmentation-and-recognition-for-vehicle-license-plate/

This class just contains the neural network to analyze signatures. It will process images, map them to arrays, 
learn through backpropogation, and save progress to a file.

Learning Process Step by Step:
1. Take random image from train and its corresponding value from written_name_train
2. Segment the image into individual characters
3. For each character, guess what that letter is
4. Once all letters have been evaluated, return the string it spells out
5. Then, we can compare the correct string to the evaluated one, and use backpropogation
'''
class Signature():

    slope = 1
    dataLength = 0
    INPUT_LENGTH = 900
    dataRows = []
    network = []
    errors = []
    alpha = 0.5
    nodesPerLayer = [40, 40, 26]
    numberHiddenLayers = len(nodesPerLayer)
    dataPath = 'C:/Datasets/signatures/'
    csvPath = os.getcwd()+'/answerkey.csv'
    weightPath = os.getcwd()+'/weights'
    CHEAT = True

    def __init__(self):

        self.init_network(False)
        self.init_training_data()

        for i in range(1000):
            all_outputs, guess, spelling = self.evaluate_random_sample()
            print("Name: "+str(spelling)+"; Guess: "+"".join(guess))
            for i, output in enumerate(all_outputs):
                correct_output = self.correct_letter_to_array(spelling[i])

                self.backpropogation(output, correct_output)

    # Handles all of the initializations
    # read - Boolean. If true, reads in weights from file
    def init_network(self, read):
        # There are INPUT_LENGTH input nodes.
        # Each layer of n nodes with m inputs is a 2-tuple containing an n by m weight matrix and an n by 1 bias array
        # REMINDER : np is always (rows, columns) so a matrix with shape (1, 3) looks like [0, 0, 0]

        # init input layer
        hidden_layers = []
        error_layers = []
        if read:
            pass
        else:
            prev_outputs = self.INPUT_LENGTH
            for i in range(self.numberHiddenLayers):
                # inserts random numbers from [-1, 1)
                weightmat = ((np.random.rand(self.nodesPerLayer[i], prev_outputs + 1)-0.5)*2).tolist()
                errormat = np.zeros(self.nodesPerLayer[i]).tolist()

                hidden_layers.append(weightmat)
                error_layers.append(errormat)
                prev_outputs = self.nodesPerLayer[i]

        self.network = hidden_layers
        self.errors = error_layers
        return

    # Should return the outputs of all nodes
    def evaluate_random_sample(self):

        file, spelling = self.grab_sample()
        binary_image = self.clean_image(file)
        letters = self.segment(binary_image)

        # if the program segments improperly, then throw away this sample and grab a new one.
        if self.CHEAT:
            while len(letters) != len(spelling):
                file, spelling = self.grab_sample()
                binary_image = self.clean_image(file)
                letters = self.segment(binary_image)

        guess = []
        all_outputs = []
        for i, letter in enumerate(letters):
            input_list = letter
            outputs = self.get_network_output(input_list)
            # This converts the output layer into a character (Highest value in the output array is the official guess)
            index_max = max(range(len(outputs[-1])), key=outputs[-1].__getitem__)
            guess.append(chr(index_max+97))
            all_outputs.append(outputs)

            # error = self.error(outputs[-1], spelling[i])
        return all_outputs, guess, spelling.lower()

    # Takes the final output of the network and sees how close it is
    def correct_letter_to_array(self, answer):
        # construct an array based on the answer
        actual = [0]*26
        try:
            actual[ord(answer) - 97] = 1
        except:
            print(answer)
        return actual

    # Perform backpropogation on the NN
    # the input array is stored as the first element of output_list
    def backpropogation(self, output_list, expected_list):
        # a = self.network[::-1] network in reverse

        # The error list has the same shape as the neural network, so we just have to populate it with errors
        # We will do the output nodes first (their error is just how far off they were from the target)

        # Reverse the network for easier indexing
        for i, layer in enumerate(self.network[::-1]):
            if i == 0:
                self.errors[self.numberHiddenLayers - 1] = ((np.array(expected_list) - output_list[-1])).tolist()
            else:
                errorvec = []
                for j, node in enumerate(layer):
                    error = 0.0
                    # Loop through connected nodes in the previous layer
                    for k, connected_node in enumerate(self.network[self.numberHiddenLayers - i]):
                        error += (connected_node[j] * self.errors[self.numberHiddenLayers - i][k])
                    errorvec.append(error)
                self.errors[self.numberHiddenLayers - i - 1] = errorvec
            for j, a in enumerate(layer):
                self.errors[self.numberHiddenLayers - i - 1][j] *= self.sigmoid_derivative(output_list[self.numberHiddenLayers - i][j])

        for i, layer in enumerate(self.network):
            for j, node in enumerate(layer):
                for k, weight in enumerate(node):
                    delta = self.errors[i][j]
                    out = output_list[i+1][j]
                    temp = self.alpha*delta*out

                    self.network[i][j][k] += self.alpha*delta*out

        return

    # Given x, return f'(x) where f is the sigmoid function
    def sigmoid_derivative(self, out):
        return out * (1 - out) / self.slope

    # Initialize the dataset (Get number of images stored)
    def init_training_data(self):
        with open(self.csvPath, newline='') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                self.dataRows.append(row)

            self.dataLength = reader.line_num

    # Returns the image filename and correct spelling as a tuple (Filename, Identity)
    def grab_sample(self):
        a = random.randint(0, self.dataLength)
        return self.dataRows[a]

    # Applies filters to the image to make it readable
    # Result should be a black and white image (np array)
    # TODO: tweak settings to get a cleaner image
    # TODO: cut out non-handwriting text
    # ndimage has interpolation tools
    def clean_image(self, image_file_name):
        blur_rad = 0.4
        threshold = 25

        im = Image.open(self.dataPath + image_file_name)
        # im.show()
        altered = ImageOps.invert(ImageOps.grayscale(im))
        altered = altered.filter(ImageFilter.GaussianBlur(radius=blur_rad))
        im_arr = np.array(altered)
        im_bin = (im_arr > threshold)*255
        # final = Image.fromarray(im_bin)
        return im_bin

    # Returns an array of all the images of individual characters.
    def segment(self, image):
        h, w = image.shape

        # all connected pieces are labelled
        connected, nr_objects = ndimage.label(image)

        # TODO
        # if two connected objects have similar x coordinates, they are probably a part of the same character, so let's
        # combine them


        # for each object, get its bounding box
        a = ndimage.find_objects(connected)
        letters = []
        locations = []

        '''
        for i, box in enumerate(a):
            for box2 in enumerate(a[i:]):
                pass
        '''

        # if a box is hella long, remove it
        # if a box is not letter shaped, remove it
        # if a region goes off the screen, remove it because it's probably something unimportant
        # some of these crops are really bad, so if that region takes up more than 40% of the height, then it's fine
        for i, box in enumerate(a):
            box_width = abs(box[1].stop - box[1].start)
            box_height = abs(box[0].stop - box[0].start)

            if (box_width / w > 0.1):
                continue
            if (box_height / h < 0.2):
                continue

            if (box[1].stop * box[0].start > 1 and box[1].start <= w - 1 and box[0].stop <= h - 1) or \
                    abs(box[0].stop - box[0].start) / h > 0.5:
                locations.append(box)
                # I am going to pad the input array after ravel-ing it. The order definitely matters,
                # but I will try this first

                # if a pixel is non-zero, change it to 1

                cropped_letter = ((connected[box] > 0)*1).ravel()
                padded = np.pad(cropped_letter, (0, 900-len(cropped_letter)), mode='constant')
                letters.append(padded)

                '''
                cropped_letter = ((connected[box] > 0)*1)
                padded = np.pad(cropped_letter, ((0, 40-len(cropped_letter)), (0, 40-len(cropped_letter[0]))), mode='constant')
                letters.append(padded.ravel())
                '''



        # Some code to draw the regions found
        '''
        im = Image.fromarray(image)
        draw = ImageDraw.Draw(im)
        for box in locations:
            draw.rectangle((box[1].stop, box[0].start, box[1].start, box[0].stop), None, 255, 1)
        im.show()
        print(len(locations))
        '''
        # Now that we have a list of the regions, we can take those parts of the image and feed them into the network
        return letters

    # Using an input array, calculate the network's output. Should return the outputs of each layer
    # inputArray should be a numpy array
    def get_network_output(self, input_array):
        outputs = []
        outputs.append(input_array)
        for i, layer in enumerate(self.network):
            outputs.append( self.eval_layer(outputs[i], layer) )
        return outputs

    # Given a layer of nodes and input to those nodes, return the output array
    def eval_layer(self, input_array, layer):
        # a is the weight matrix for the layer and b is the bias vector
        temp = np.array(layer)
        a = temp[:, :-1]
        b = temp[:, -1]
        out = self.activation( np.dot(a, input_array) + b )
        return out

    def write_weights_to_file(self):
        pass

    def read_weights_from_file(self):
        pass

    # input should be a numpy array
    def activation(self, i):
        # return 1/(1+m.exp(-(transpose)))
        return 1/(1+np.exp(-i/self.slope))