from game_of_go.dlgo.gotypes import Point


# an eye is an empty point where all
# adjacent points and at least three out of four diagonally adjacent points are filled with
#    friendly stones.

def is_point_an_eye(board, point, color):
    if board.get(point) is not None:  # An eye is an empty point.
        return False
    for neighbor in point.neighbors():  # All adjacent points must contain friendly stones.
        if board.is_on_grid(neighbor):
            neighbor_color = board.get(neighbor)
            if neighbor_color != color:
                return False
    friendly_corners = 0  # We must control three out of four corners if the point is in the middle
    # of the board; on the edge, you must control all corners.
    off_board_corners = 0
    corners = [
        Point(point.row - 1, point.col - 1),
        Point(point.row - 1, point.col + 1),
        Point(point.row + 1, point.col - 1),
        Point(point.row + 1, point.col + 1),
    ]
    for corner in corners:
        if board.is_on_grid(corner):
            corner_color = board.get(corner)
            if corner_color == color:
                friendly_corners += 1
            else:
                off_board_corners += 1
    if off_board_corners > 0:
        return off_board_corners + friendly_corners == 4  # Point is on the edge or corner.
    return friendly_corners >= 3  # Point is in the middle.
