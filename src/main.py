from PokerBots import CallingPlayer
from PokerBots import GamblingPlayer
from PokerBots import Game
from PokerBots import RandomPlayer

from llm_player import LLMPlayer

if __name__ == "__main__":
    # Define three vanila players
    player1 = GamblingPlayer(name="Gambler")
    player2 = CallingPlayer(name="Caller")
    player3 = RandomPlayer(name="Random")
    player4 = LLMPlayer(name="gpt-4.1-nano-2025-04-14")

    game = Game(players=[player1, player2, player3, player4], initial_stack=30_000)

    game.play_round(verbose=True)
