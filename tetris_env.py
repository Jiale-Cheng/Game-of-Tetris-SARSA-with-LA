import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

WIDTH = 6


class TetrisEnv:
    def __init__(self, seed=None):
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)
        self.game_area = 0
        self.piece = (self.rng.integers(low=0, high=4) << WIDTH) + self.rng.integers(low=0, high=3) + 1
        self.height = 0

    # Transform the piece representation from 2*WIDTH bits to 4 bits
    @staticmethod
    def piece_transform(piece):
        low = piece & 3
        high = (piece & (3 << WIDTH)) >> (WIDTH - 2)
        return low + high

    def reset(self):
        self.game_area = 0
        self.piece = (self.rng.integers(low=0, high=4) << WIDTH) + self.rng.integers(low=0, high=3) + 1
        self.height = 0
        return self.game_area, self.piece_transform(self.piece)

    def get_state(self):
        return self.game_area, self.piece_transform(self.piece)

    def get_height(self):
        return self.height

    @staticmethod
    def rotate(p, rotation):
        # rotate the piece clockwise
        while rotation > 0:
            q = p >> WIDTH
            p = (2 if p & 1 != 0 else 0) + (2 << WIDTH if p & 2 != 0 else 0) + (1 << WIDTH if q & 2 != 0 else 0) + (
                1 if q & 1 != 0 else 0)
            rotation = rotation - 1
        if p % (1 << WIDTH) == 0:
            p >>= WIDTH
        return p

    def step(self, position, rotation):
        assert (0 <= round(position) <= WIDTH - 2) and (
                    0 <= round(rotation) <= 3), "Error: The action input is invalid!"
        position = round(position)
        rotation = round(rotation)
        piece = self.rotate(self.piece, rotation)  # rotate the piece
        piece = piece << position  # move to the position
        # drop down the piece
        while (piece & self.game_area) or ((piece << WIDTH) & self.game_area):
            piece = piece << WIDTH
        t = piece | self.game_area

        # if the top row is full, then remove the row
        if (t & (((1 << WIDTH) - 1) << WIDTH)) == (((1 << WIDTH) - 1) << WIDTH):
            t = (t & ((1 << WIDTH) - 1)) | ((t >> (2 * WIDTH)) << WIDTH)

        # if the bottom row is full, then remove the row
        if (t & ((1 << WIDTH) - 1)) == ((1 << WIDTH) - 1):
            t >>= WIDTH

        self.game_area = t
        self.piece = (self.rng.integers(low=0, high=4) << WIDTH) + self.rng.integers(low=0, high=3) + 1

        loss = 0
        while (self.game_area >> (2 * WIDTH)) != 0:
            self.game_area = self.game_area >> WIDTH
            loss = loss + 1
        assert (self.game_area < (1 << 2 * WIDTH)), "Unknown Error!"

        self.height = self.height + loss
        reward = -loss

        return self.game_area, self.piece_transform(self.piece), reward

    # You can use this function to visualize the game
    def visualize(self, action_pos, action_rot):
        """
        Visualize the game
        Args:
            action_pos: a_{pos}, the position where the agent puts the next piece down in the current step
            action_rot: a_{rot}, clockwise rotation of the piece before putting the piece down in the current step
        """

        # borderlines of the plot
        xmin = 0
        xmax = 6
        ymin = 0
        ymax = 6

        # start plotting the game
        state_game, _ = self.get_state()

        upper_row = bin(state_game >> 6)
        upper_row = upper_row[2:].zfill(6)

        lower_row = bin(state_game & 63)
        lower_row = lower_row[2:].zfill(6)

        fig, ax = plt.subplots()

        y_lower_left_corner = 0
        x_lower_left_corner = 0

        for i in range(6):
            s = lower_row[i]
            if s == '1':
                ax.add_patch(Rectangle((x_lower_left_corner, y_lower_left_corner), 1, 1, color="black"))
            x_lower_left_corner += 1

        y_lower_left_corner += 1
        x_lower_left_corner = 0
        for i in range(6):
            s = upper_row[i]
            if s == '1':
                ax.add_patch(Rectangle((x_lower_left_corner, y_lower_left_corner), 1, 1, color="black"))
            x_lower_left_corner += 1

        # start plotting the piece
        y_lower_left_corner += 1
        state_piece = self.rotate(self.piece, action_rot) << action_pos

        upper_row_piece = bin(state_piece >> 6)
        upper_row_piece = upper_row_piece[2:].zfill(6)

        lower_row_piece = bin(state_piece & 63)
        lower_row_piece = lower_row_piece[2:].zfill(6)

        x_lower_left_corner = 0
        for i in range(6):
            s = lower_row_piece[i]
            if s == '1':
                ax.add_patch(Rectangle((x_lower_left_corner, y_lower_left_corner), 1, 1, color="red"))
            x_lower_left_corner += 1

        x_lower_left_corner = 0
        y_lower_left_corner += 1

        for i in range(6):
            s = upper_row_piece[i]
            if s == '1':
                ax.add_patch(Rectangle((x_lower_left_corner, y_lower_left_corner), 1, 1, color="red"))
            x_lower_left_corner += 1

        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        plt.xlabel("X - position")
        plt.ylabel("Height")
        plt.title("Tetris Grid")
        plt.grid()
        black_patch = mpatches.Patch(color='black', label='Game Area')
        red_patch = mpatches.Patch(color='red', label='Rotated Piece')
        plt.legend(handles=[red_patch, black_patch])
        plt.show()
