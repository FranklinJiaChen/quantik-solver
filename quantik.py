import numpy as np

class Quantik:
    """
    A class representing the Quantik board game.

    Attributes:
    - board (np.ndarray): A 4x4 numpy array representing the game board.
    - last_move (list): A list containing [row, column] of the last move made.
    - current_player (int): An integer representing the current player (1 or 2).

    Methods:
    - print_board(): Prints the current state of the board.
    """
    def __init__(self):
        self.board = [[0 for _ in range(4)] for _ in range(4)]  # Initialize a 4x4 board
        self.board = np.array(self.board)
        self.last_move = [-1, -1] # row, col (0-indexed)
        self.current_player = 1  # Player 1 starts first


    def print_board(self):
        for row in self.board:
            print(' '.join(map(lambda x: str(x).rjust(-2), row)))

    def place_piece(self, piece, row, col):
        if self.board[row][col] == 0:
            if self.current_player == 1 and piece in self.player1_pieces:
                self.board[row][col] = piece
                self.current_player = 2  # Switch turn to player 2
                return True
            elif self.current_player == 2 and piece in self.player2_pieces:
                self.board[row][col] = piece
                self.current_player = 1  # Switch turn to player 1
                return True
            else:
                print("Invalid move: Piece not owned by current player.")
        else:
            print("Invalid move: Cell is already occupied.")
        return False

# Example usage:
game = Quantik()
game.place_piece(1, 0, 0)  # Player 1 places piece 1 at (0, 0)
game.place_piece(-3, 1, 1)  # Player 2 places piece -3 at (1, 1)
game.print_board()  # Print the current board state
