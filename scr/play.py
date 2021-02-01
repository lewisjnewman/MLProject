from NNPlayer import NNPlayer
from NeuralNetwork import NeuralNetwork
from CompromiseGame import CompromiseGame,RandomPlayer,GreedyPlayer,SmartGreedyPlayer,DeterminedPlayer

import traceback


def main():
    p1 = NNPlayer()
    p1.nn = NeuralNetwork.load_json("best_player.json")

    p2 = RandomPlayer()

    game = CompromiseGame(p1, p2, 30, 10)

    wins = 0
    games = 1000

    neural_network_scores = []
    opponent_scores = []

    try:
        for i in range(games):
            game.resetGame()
            results = game.play()

            neural_network_scores.append(results[0])
            opponent_scores.append(results[1])

            if results[0] > results[1]:
                wins += 1

            if i % 100 == 0:
                print(f"Games Played: {i}")

    except:
        traceback.print_exc()

    print(f"Wins = {wins}")
    print(f"Games Played = {games}")
    print(f"Winning Percentage = {round(wins/games*100)}")
    print(f"Neural Network Player Average Points = {sum(neural_network_scores)/len(neural_network_scores)}")
    print(f"Opponent Average Points = {sum(opponent_scores)/len(opponent_scores)}")

if __name__ == "__main__":
    main()
