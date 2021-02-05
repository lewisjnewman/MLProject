import matplotlib.pyplot as plt
import numpy as np
import sys

with open(sys.argv[1]) as datafile:
    datalines = datafile.read().splitlines()

for i in range(len(datalines)):
    datalines[i] = datalines[i].split(",")

data = list(zip(*datalines))

gen_count = list(map(int,data[0]))
max_scores = list(map(float,data[1]))
mean_scores = list(map(float,data[2]))
min_scores = list(map(float,data[3]))
max_winrates = list(map(float,data[4]))
mean_winrates = list(map(float,data[5]))
min_winrates = list(map(float,data[6]))


fig, axs = plt.subplots(2)

axs[0].plot(gen_count, max_scores, label="Max Scores")
axs[0].plot(np.convolve(max_scores, np.ones(25)/25, mode='valid'), label="Windowed Max Average")
axs[0].plot(gen_count, mean_scores, label="Mean Scores")
axs[0].plot(np.convolve(mean_scores, np.ones(25)/25, mode='valid'), label="Windowed Mean Average")
axs[0].plot(gen_count, min_scores, label="Min Scores")
axs[0].plot(np.convolve(min_scores, np.ones(25)/25, mode='valid'), label="Windowed Min Average")
axs[0].grid(True,which="both",axis="y")
axs[0].legend()

axs[1].plot(gen_count, max_winrates, label="Max winrates")
axs[1].plot(np.convolve(max_winrates, np.ones(25)/25, mode='valid'), label="Windowed Max Average")
axs[1].plot(gen_count, mean_winrates, label="Mean winrates")
axs[1].plot(np.convolve(mean_winrates, np.ones(25)/25, mode='valid'), label="Windowed Mean Average")
axs[1].plot(gen_count, min_winrates, label="Min winrates")
axs[1].plot(np.convolve(min_winrates, np.ones(25)/25, mode='valid'), label="Windowed Min Average")
axs[1].grid(True,which="both",axis="y")
axs[1].legend()

plt.show()
