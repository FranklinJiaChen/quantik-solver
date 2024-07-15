import unittest
from quantik import is_terminal

class TestIsTerminal(unittest.TestCase):

    def test_terminal_row(self):
        board = (2, 4, -3, -1,
                 0, 0, 0, 0,
                 0, 0, 0, 0,
                 0, 0, 0, 0)
        pos = 3
        self.assertTrue(is_terminal(board, pos))

    def test_terminal_column(self):
        board = (2, 0, -3, -1,
                 3, 0, 0, 0,
                 -1, 0, 0, 0,
                 -4, 0, 0, 0)
        pos = 0
        self.assertTrue(is_terminal(board, pos))

    def test_terminal_region(self):
        board = (1, 2, 3, 0,
                 -3, -4, 0, 0,
                 0, 0, 0, 0,
                 0, 0, 0, 0)
        pos = 0
        self.assertTrue(is_terminal(board, pos))

    def test_non_terminal(self):
        board = (0, 0, 0, 0,
                 0, 0, 0, 0,
                 0, 0, 0, 0,
                 0, 0, 0, 0)
        pos = 15
        self.assertFalse(is_terminal(board, pos))

if __name__ == '__main__':
    unittest.main()