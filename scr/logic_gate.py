from NeuralNetwork import *
from random import randint
import multiprocessing
import copy
import numpy as np

NETWORK_SHAPE=[5,5,2]

#TRAINING PARAMETERS
POPULATION=30
NUM_GENERATIONS=1000
MUTATION_CHANCE=0.01
MUTATION_LIMIT=2.5

def play(nn:NeuralNetwork, games=50):
    wins = 0

    for i in range(games):
        val = [randint(0, 1), randint(0,1), randint(0, 1), randint(0,1), randint(0,1)]
        expected = val[0] & val[1] | val[2] | val[3] ^ val[4]

        outputs = nn.forward(val)

        actual = np.argmax(outputs)
        if expected == actual:
            wins+=1

    return wins/games

def breed_and_mutate(players, scores):
    players_with_scores = tuple(zip(players, scores))
    players_with_scores = sorted(players_with_scores, key=lambda x: x[1])

    # Copy the top 1/3 of players
    new_players = copy.deepcopy(players_with_scores[2*len(players_with_scores)//3:])

    # Remove the bottom 1/3 of players
    players_with_scores = players_with_scores[len(players_with_scores)//3:]

    # Mutate the copies of the best players
    for i in range(len(new_players)):
        player = new_players[i][0]
        c = player.GetChromosomes()
        c = mutate_replace_chromosomes(c, MUTATION_CHANCE, MUTATION_LIMIT)
        player.SetFromChromosomes(c)
        new_players[i] = (player, new_players[i][1])

    # Add the mutated copies back in
    players_with_scores += new_players

    # Blend the middlish players with the original top players
    # Then replace the original middle players with these new blended players
    for i in range(len(players_with_scores)//3):
        p1 = players_with_scores[i][0]
        p2 = players_with_scores[i + len(players_with_scores)//3][0]
        p3_chromo = merge_chromosomes(p1.GetChromosomes(), p2.GetChromosomes())
        players_with_scores[i][0].SetFromChromosomes(p3_chromo)

    return list(map(lambda x: x[0],players_with_scores))

def main():

    with open("logic_gate_results.csv", "w") as datafile:
        pool = multiprocessing.Pool(processes=32)

        players = []
        for _ in range(POPULATION):
            players.append(NeuralNetwork(NETWORK_SHAPE))

        for g in range(NUM_GENERATIONS):
            scores = list(pool.map(play, players))
            players = breed_and_mutate(players, scores)

            print(f"\nGeneration {g+1}")
            print(f"Top Score = {max(scores)}")
            print(f"Mean Score = {sum(scores)/len(scores)}")
            print(f"Bottom Score = {min(scores)}")

            datafile.write(f"{g},{max(scores)},{sum(scores)/len(scores)},{min(scores)}\n")
            datafile.flush()

if __name__ == "__main__":
    main()
