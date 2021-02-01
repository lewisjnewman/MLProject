from NNPlayer import NNPlayer
from NeuralNetwork import NeuralNetwork
from CompromiseGame import CompromiseGame,RandomPlayer,GreedyPlayer,SmartGreedyPlayer,DeterminedPlayer

import traceback


def main():
    p1 = SmartGreedyPlayer()
    p2 = RandomPlayer()

    game = CompromiseGame(p1, p2, 30, 10)

    wins = 0
    games = 1000

    smart_greedy_scores = []
    random_player_scores = []

    try:
        for i in range(games):
            game.resetGame()
            results = game.play()

            smart_greedy_scores.append(results[0])
            random_player_scores.append(results[1])

            if results[0] > results[1]:
                wins += 1

            if i % 100 == 0:
                print(f"Games Played: {i}")

    except:
        traceback.print_exc()

    print(f"Smart Greedy Wins = {wins}")
    print(f"Games Played = {games}")
    print(f"Winning Percentage = {round(wins/games*100)}")
    print(f"Random Player Average Points = {sum(random_player_scores)/len(random_player_scores)}")
    print(f"Smart Greedy Average Points = {sum(smart_greedy_scores)/len(smart_greedy_scores)}")

if __name__ == "__main__":
    main()
