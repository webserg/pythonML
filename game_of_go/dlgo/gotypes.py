import enum
from collections import namedtuple


class Player(enum.Enum):
    black = 1
    white = 2

    # After a player places a stone, you can switch the color by calling the other method
    @property
    def other(self):
        return Player.black if self == Player.white else Player.white


# to represent coordinates on the board, tuples are an obvious choice
class Point(namedtuple('Point', 'row col')):
    def neighbors(self):
        return [
            Point(self.row - 1, self.col),
            Point(self.row + 1, self.col),
            Point(self.row, self.col - 1),
            Point(self.row, self.col + 1),
        ]
