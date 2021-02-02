from NNPlayer import NNPlayer
from NeuralNetwork import *
from CompromiseGame import CompromiseGame,RandomPlayer,GreedyPlayer,SmartGreedyPlayer,DeterminedPlayer
import multiprocessing
import copy
import random
import sys

#TRAINING PARAMETERS
POPULATION=250
NUM_GENERATIONS=2000
MUTATION_CHANCE=0.05
MUTATION_LIMIT=1

def calculate_fitness(result):
    average_player_score = result[0]
    average_score_difference = result[1]
    winrate = result[2]

    return winrate
    #return average_player_score
    #return average_score_difference

def play(nnplayer, games=50):
    p2 = RandomPlayer()

    game = CompromiseGame(nnplayer, p2, 30, 10)

    wins = 0
    player_scores = []
    score_differences = []

    for i in range(games):
        game.resetGame()
        results = game.play()

        if results[0] > results[1]:
            wins += 1

        player_scores.append(results[0])
        score_differences.append(results[0] - results[1])

    return (sum(player_scores)/len(player_scores), sum(score_differences)/len(score_differences),wins/games)

def breed_and_mutate(players, scores):
    players_with_scores = tuple(zip(players, scores))

    # Order the players by their fitness scores
    players_with_scores = sorted(players_with_scores, key=lambda x: x[1])

    # Copy the top 10% of players
    new_players = copy.deepcopy(players_with_scores[9*len(players_with_scores)//10:])

    # Remove the bottom 30% of players
    players_with_scores = players_with_scores[3*len(players_with_scores)//10:]

    # Mutate the copies of the best players
    for i in range(len(new_players)):
        player = new_players[i][0]
        c = player.nn.GetChromosomes()
        c = mutate_adjust_chromosomes(c, MUTATION_CHANCE, MUTATION_LIMIT)
        player.nn.SetFromChromosomes(c)
        new_players[i] = (player, new_players[i][1])


    merged_players = []

    # Fill the the missing population with mergers between random players
    while len(players_with_scores) + len(merged_players) + len(new_players) < POPULATION:

        # Choose 2 random players from the top 20%
        p1 = random.choice(players_with_scores[8*len(players_with_scores)//10:])[0]
        p2 = random.choice(players_with_scores[8*len(players_with_scores)//10:])[0]

        # Create a new player by merging their chromosomes
        p3_chromo = merge_chromosomes(p1.nn.GetChromosomes(), p2.nn.GetChromosomes())
        p3 = copy.deepcopy(p1)
        p3.nn.SetFromChromosomes(p3_chromo)

        # Add this new player to the list of new players
        merged_players.append((p3,))

    # Add the mutated copies back in
    players_with_scores += new_players

    # Add the new merged players in
    players_with_scores += merged_players

    """
    # Blend the middlish players with the original top players
    # Then replace the original middle players with these new blended players
    for i in range(len(players_with_scores)//3):
        p1 = players_with_scores[i][0]
        p2 = players_with_scores[i + len(players_with_scores)//3][0]
        p3_chromo = merge_chromosomes(p1.nn.GetChromosomes(), p2.nn.GetChromosomes())
        players_with_scores[i][0].nn.SetFromChromosomes(p3_chromo)
    """

    return list(map(lambda x: x[0],players_with_scores))

def main():

    best_player = None
    best_player_fitness = None


    with open("results.csv", "w") as datafile:
        pool = multiprocessing.Pool(processes=60)

        players = []
        if len(sys.argv) == 1:
            # Generate Players from random
            for _ in range(POPULATION):
                players.append(NNPlayer())
        else:
            # Generate Players from existing neural network
            p = NNPlayer()
            p.nn = NeuralNetwork.load_json(sys.argv[1])
            for _ in range(POPULATION):
                players.append(copy.deepcopy(p))

            # Run a mutation straight away so that we don't have N copies of the exact same player
            # Leave 1 player unmutated
            for i in range(1, len(players)):
                player = players[i]
                c = player.nn.GetChromosomes()
                c = mutate_adjust_chromosomes(c, MUTATION_CHANCE, MUTATION_LIMIT)
                player.nn.SetFromChromosomes(c)
                players[i] = player


        for g in range(NUM_GENERATIONS):
            print(f"\nGeneration {g+1}")

            results = list(pool.map(play, players))

            player_scores = list(map(lambda x: x[0], results))
            score_differences = list(map(lambda x: x[1], results))
            winrates = list(map(lambda x: x[2], results))

            fitnesses = list(map(calculate_fitness, results))

            players = breed_and_mutate(players, fitnesses)

            print(f"Top Score = {max(player_scores)}")
            print(f"Mean Score = {sum(player_scores)/len(player_scores)}")
            print(f"Bottom Score = {min(player_scores)}")
            print(f"Top Winrate = {max(winrates)}")
            print(f"Mean Winrate = {sum(winrates)/len(winrates)}")
            print(f"Bottom Winrate = {min(winrates)}")

            datafile.write(f"{g},{max(player_scores)},{sum(player_scores)/len(player_scores)},{min(player_scores)},{max(winrates)},{sum(winrates)/len(winrates)},{min(winrates)}\n")
            datafile.flush()

            pws = tuple(zip(players, fitnesses))

            if best_player == None:
                best_player = max(pws, key=lambda x: x[1])[0]
                best_player_fitness = max(pws, key=lambda x: x[1])[1]
            else:
                generation_best = max(pws, key=lambda x: x[1])
                if generation_best[1] >= best_player_fitness:
                    best_player = generation_best[0]
                    best_player_fitness = generation_best[1]

    best_player.nn.save_json("best_player.json")

if __name__ == "__main__":
    main()
