from CompromiseGame import AbstractPlayer
from NeuralNetwork import NeuralNetwork

import numpy as np

# [Input Layer] + [Hidden Layers] + [Outputs]

NN_SHAPE = [54]+[32,32]+[9]

class NNPlayer(AbstractPlayer):
    def __init__(self):
        self.nn = NeuralNetwork(NN_SHAPE)

        self.log = ""

    def _write_log(self, string):
        """Log Writer for Debugging the class/neural network"""
        #print(string)
        self.log += string + "\n"

    def play(self, myState, oppState, myScore, oppScore, turn, length, nPips):
        """Overrides AbstractPlayer.play"""

        self._write_log(f"Turn Number: {turn}")
        self._write_log(f"Player Score: {myScore}")
        self._write_log(f"Oppenent Score: {oppScore}")

        # ######################
        # Inputs are:
        #   myState = 3x3x3 array of ints
        #   oppState = 3x3x3 array of ints
        #   myScore = int
        #   oppScore = int
        #   turn = int
        #   length = int
        #   nPips = int
        # ######################
        # Need to return an array of length == nPips
        # each element of the array is a sub array of length 3
        # each element of the sub array is an intefer of the set (0,1,2)

        my_state_flat = []
        opp_state_flat = []

        for i in range(3):
            for j in range(3):
                my_state_flat += myState[i][j]

        for i in range(3):
            for j in range(3):
                opp_state_flat += oppState[i][j]

        input_data = my_state_flat + opp_state_flat

        input_data = list(map(float,input_data))

        self._write_log(f"NN Input = {input_data}")

        output_data = self.nn.forward(input_data).flatten().tolist()

        self._write_log(f"NN Output = {output_data}")

        d1 = output_data[0:3]
        d2 = output_data[3:6]
        d3 = output_data[6:9]
        decision = [np.argmax(d1), np.argmax(d2), np.argmax(d3)]

        self._write_log(f"Decision = {decision}")

        return decision
