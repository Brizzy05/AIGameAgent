# Student agent: Add your own agent here
# Minimax depth limit and heuristic
import sys
from copy import deepcopy
import numpy as np
from agents.agent import Agent
from store import register_agent


@register_agent("minimaxH_agent")
class MinimaxHAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(MinimaxHAgent, self).__init__()
        self.name = "MinimaxHAgent"

        # Moves (Up, Right, Down, Left)
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        # Opposite Directions
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        # dummy return

        # bfs to find closest pos to adv
        print("\n")
        board_size = len(chess_board)

        move = self.minimax(True, my_pos, adv_pos, chess_board, board_size, 2)

        r, x, d = move["move"]

        return (r, x), d

    def valid_move(self, chess_board, my_pos, max_step, board_size, adv_pos):
        # gen all possible moves from starting position
        move_list = []
        r, c = my_pos
        r2, c2 = adv_pos
        for i in range(board_size):
            for j in range(board_size):
                new_pos = i, j
                x, y = new_pos
                for k in range(4):
                    if (self.check_valid_step(np.array([r, c]), np.array([x, y]), k, max_step, chess_board,
                                              np.array([r2, c2]))):
                        move_list.append((x, y, k))

        return move_list

    def minimax(self, isMaximizing, my_pos, adv_pos, chess_board, board_size, depth):
        max_step = (board_size + 1) // 2
        move_list = self.valid_move(chess_board, my_pos, max_step, board_size, adv_pos)

        if depth == 0:
            move_listb = self.valid_move(chess_board, adv_pos, max_step, board_size, my_pos)
            count = self.count_path(chess_board, move_list, move_listb, my_pos, adv_pos)
            return {"move": None, "score": count}

        # check if endgame
        end_game, p0_score, p1_score = self.check_endgame(board_size, chess_board, my_pos, adv_pos)

        if end_game:
            return {"move": None, "score": p0_score - p1_score if isMaximizing else p1_score - p0_score}

        # if we are the maximizing player
        if isMaximizing:
            best = {"move": None, "score": -100000}
        else:
            best = {"move": None, "score": 100000}

        for mv in move_list:
            r, c, d = mv
            new_pos = (r, c)
            self.set_barrier(r, c, d, chess_board)
            if isMaximizing:
                sim_score = self.minimax(False, my_pos=adv_pos, adv_pos=new_pos, chess_board=chess_board,
                                         board_size=board_size, depth=depth - 1)
            else:
                sim_score = self.minimax(True, my_pos=adv_pos, adv_pos=new_pos, chess_board=chess_board,
                                         board_size=board_size, depth=depth - 1)

            # undo
            self.undo_barrier(r, c, d, chess_board)

            # possible optimal move

            sim_score["move"] = mv
            # print("after", sim_score)

            if isMaximizing:
                if sim_score["score"] > best["score"]:
                    best = sim_score
            else:
                if sim_score["score"] < best["score"]:
                    best = sim_score

        return best

    @staticmethod
    def count_path(chess_board, move_lista, move_listb, my_pos, adv_pos):
        # calculate the number of walls reachable by both players

        max_count = 0
        min_count = 0
        r1, c1 = my_pos
        r2, c2 = adv_pos

        for r, c, d in move_lista:
            if chess_board[r, c, d]:
                max_count += 1

        for r, c, d in move_listb:
            if chess_board[r, c, d]:
                min_count += 1

        count = max_count - min_count
        return count

    def check_endgame(self, board_size, chess_board, my_pos, adv_pos):
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        # Union-Find
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                        self.moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(my_pos)
        p1_r = find(adv_pos)
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            player_win = 0
            win_blocks = p0_score
        elif p0_score < p1_score:
            player_win = 1
            win_blocks = p1_score
        else:
            player_win = -1  # Tie

        return True, p0_score, p1_score

    def check_valid_step(self, start_pos, end_pos, barrier_dir, max_step, chess_board, adv_pos):
        """
        Check if the step the agent takes is valid (reachable and within max steps).

        Parameters
        ----------
        start_pos : tuple
            The start position of the agent.
        end_pos : np.ndarray
            The end position of the agent.
        barrier_dir : int
            The direction of the barrier.
        """
        
        # Endpoint already has barrier or is boarder
        r, c = end_pos
        if chess_board[r, c, barrier_dir]:
            return False
        if np.array_equal(start_pos, end_pos):
            return True

        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == max_step:
                break
            for dir, move in enumerate(self.moves):
                if chess_board[r, c, dir]:
                    continue

                next_pos = cur_pos + move
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached

    def set_barrier(self, r, c, dir, chess_board):
        # Set the barrier to True
        chess_board[r, c, dir] = True
        # Set the opposite barrier to True
        move = self.moves[dir]
        chess_board[r + move[0], c + move[1], self.opposites[dir]] = True

    def undo_barrier(self, r, c, dir, chess_board):
        # Set the barrier to True
        chess_board[r, c, dir] = False
        # Set the opposite barrier to True
        move = self.moves[dir]
        chess_board[r + move[0], c + move[1], self.opposites[dir]] = False
        

    