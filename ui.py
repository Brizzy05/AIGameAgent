# MatPlot Library
import matplotlib.pyplot as plt

# Constants
from constants import *

# Path Library
from pathlib import Path

# User Interface Engine
class UIEngine:

    # Initialization
    def __init__(self, grid_width=5, world=None) -> None:

        # Grid size
        self.grid_size = (grid_width, grid_width)

        # World
        self.world = world

        # Step number
        self.step_number = 0

        # Plot the figure
        plt.figure()
        # plt.axis([0, 0, 0, 10])
        plt.ion()

    # [x1,x2] -> [y1,y2]
    def plot_box(self,x,y,w,text="",set_left_wall=False,set_right_wall=False,set_top_wall=False,set_bottom_wall=False,color="silver"):

        # left wall
        plt.plot([x, x], [y, y + w], "-", lw=2, color="red" if set_left_wall else color)

        # top wall
        plt.plot([x + w, x],[y + w, y + w],"-",lw=2,color="red" if set_top_wall else color)
        
        # right wall
        plt.plot([x + w, x + w],[y, y + w],"-",lw=2,color="red" if set_right_wall else color)

        # bottom wall
        plt.plot([x, x + w], [y, y], "-", lw=2, color="red" if set_bottom_wall else color)

        # 
        if len(text) > 0:
            
            color = "black"
            
            if text == PLAYER_1_NAME:
                color = PLAYER_1_COLOR
            
            elif text == PLAYER_2_NAME:
                color = PLAYER_2_COLOR
            
            plt.text(x + w / 2,y + w / 2,text,ha="center",va="center",color="white",bbox=dict(facecolor=color, edgecolor=color, boxstyle="round"))

    # Plot the grid
    def plot_grid(self):
        for x in range(1, self.grid_size[0] * 2 + 1, 2):
            for y in range(1, self.grid_size[1] * 2 + 1, 2):
                self.plot_box(x, y, 2)

    # Plot the game boundary
    def plot_game_boundary(self):
 
        # start y=3 as the y in the range ends in 3
        self.plot_box(1, 3, self.grid_size[0] + self.grid_size[1], color="black")

    # Plot the grid with the board
    def plot_grid_with_board(self, chess_board, player_1_pos=None, player_2_pos=None, debug=False):
 
        x_pos = 0
        for y in range(self.grid_size[1] * 2 + 1, 1, -2):
            y_pos = 0
            for x in range(1, self.grid_size[0] * 2 + 1, 2):
                up_wall = chess_board[x_pos, y_pos, 0]
                right_wall = chess_board[x_pos, y_pos, 1]
                down_wall = chess_board[x_pos, y_pos, 2]
                left_wall = chess_board[x_pos, y_pos, 3]

                # Display text
                text = ""
                if player_1_pos is not None:
                    if player_1_pos[0] == x_pos and player_1_pos[1] == y_pos:
                        text += "A"
                if player_2_pos is not None:
                    if player_2_pos[0] == x_pos and player_2_pos[1] == y_pos:
                        text += "B"

                if debug:
                    text += " " + str(x_pos) + "," + str(y_pos)

                self.plot_box(x,y,2,set_left_wall=left_wall,set_right_wall=right_wall,set_top_wall=up_wall,set_bottom_wall=down_wall,text=text)

                y_pos += 1
            
            x_pos += 1

    # Fix the axis
    def fix_axis(self):
  
        # Set X labels
        ticks = list(range(0, self.grid_size[0] * 2))
        labels = [x // 2 for x in ticks]
        ticks = [x + 2 for i, x in enumerate(ticks) if i % 2 == 0]
        labels = [x for i, x in enumerate(labels) if i % 2 == 0]
        plt.xticks(ticks, labels)

        # Set Y labels
        ticks = list(range(0, self.grid_size[1] * 2))
        labels = [x // 2 for x in ticks]
        ticks = [x + 3 for i, x in enumerate(ticks) if i % 2 == 1]
        labels = [x for i, x in enumerate(reversed(labels)) if i % 2 == 1]
        plt.yticks(ticks, labels)
        
        # move x axis to top
        plt.tick_params(bottom=False, labelbottom=False, top=True, labeltop=True)
        plt.xlabel("Y Position")
        plt.ylabel("X Position", position="top")

    # Plot the text information
    def plot_text_info(self):
   
        turn = 1 - self.world.turn
        agent_0 = f"{PLAYER_1_NAME}: {self.world.p0}"
        agent_1 = f"{PLAYER_2_NAME}: {self.world.p1}"

        plt.figtext(0.15,0.1,agent_0,wrap=True,horizontalalignment="left",color=PLAYER_1_COLOR,fontweight="bold" if turn == 0 else "normal")

        plt.figtext(0.15,0.05,agent_1,wrap=True,horizontalalignment="left",color=PLAYER_2_COLOR,fontweight="bold" if turn == 1 else "normal")

        if len(self.world.results_cache) > 0:

            plt.figtext(0.4,0.1,f"Scores: A: [{self.world.results_cache[1]}], B: [{self.world.results_cache[2]}]",horizontalalignment="left")

            if self.world.results_cache[0]:

                # Handle Tie condition
                if self.world.results_cache[1] > self.world.results_cache[2]:
                    win_player = "Player A wins!"

                elif self.world.results_cache[1] < self.world.results_cache[2]:
                    win_player = "Player B wins!"
                
                else:
                    win_player = "It is a Tie!"

                plt.figtext(0.4,0.05,win_player,horizontalalignment="left",fontweight="bold",color="green")

        plt.figtext(0.7, 0.1, f"Max steps: {self.world.max_step}", horizontalalignment="left")

    # Render the board with player positions
    def render(self, chess_board, p1_pos, p2_pos, debug=False):

        plt.clf()
        
        self.plot_grid_with_board(chess_board, p1_pos, p2_pos, debug=debug)
        self.plot_game_boundary()
        self.fix_axis()
        self.plot_text_info()
        
        plt.subplots_adjust(bottom=0.2)
        plt.pause(0.1)

        if self.world.display_save:
            Path(self.world.display_save_path).mkdir(parents=True, exist_ok=True)
            plt.savefig(f"{self.world.display_save_path}/{self.world.player_1_name}_{self.world.player_2_name}_{self.step_number}.pdf")
        
        self.step_number += 1

# Start of program
if __name__ == "__main__":

    # Engine
    engine = UIEngine((5,5))

    # Render the engine
    engine.render()

    # Show the plot
    plt.show()
