# Student agent: Add your own agent here
# alpha beta depth limit and heuristic
import sys
from copy import deepcopy
import numpy as np
from agents.agent import Agent
from store import register_agent


@register_agent("alphabeta_agent")
class AlphaBetaAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(AlphaBetaAgent, self).__init__()
        self.name = "AlphaBetaAgent"

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

        move = self.alpha_beta(True, my_pos, adv_pos, chess_board, board_size, 3, -100000, 100000, 0)

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
                #if x + y <= max_step:
                for k in range(4):
                    if (self.check_valid_step(np.array([r, c]), np.array([x, y]), k, max_step, chess_board,
                                              np.array([r2, c2]))):
                        move_list.append((x, y, k))

        return self.selectMove(chess_board, move_list, adv_pos)

    def alpha_beta(self, isMaximizing, my_pos, adv_pos, chess_board, board_size, depth, alpha, beta, direction):
        max_step = (board_size + 1) // 2
        move_list = self.valid_move(chess_board, my_pos, max_step, board_size, adv_pos)

        row, col = my_pos
        row2, col2 = adv_pos

        #print(len(move_list))
        #print("History", "My_pos", my_pos, "direction", direction, "Opponent", adv_pos, "Turn", isMaximizing, alpha, beta)
        # distance between two points
        # dis = 3 / ((np.sqrt(pow((row - row2), 2) + pow((col - col2), 2))))
        #
        # if dis >= max_step:
        #     if col > board_size // 2:
        #         move_list.reverse()

        if depth == 0:
            move_listb = self.valid_move(chess_board, adv_pos, max_step, board_size, my_pos)
            count = self.heuristic(chess_board, move_list, move_listb, my_pos, adv_pos, direction, isMaximizing)
            return {"move": None, "score": count}

        # check if endgame
        end_game, p0_score, p1_score = self.check_endgame(board_size, chess_board, my_pos, adv_pos)

        if end_game:
            return {"move": None, "score": (p0_score - p1_score) * 30 if isMaximizing else (p1_score - p0_score) * 30}

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
                sim_score = self.alpha_beta(False, my_pos=adv_pos, adv_pos=new_pos, chess_board=chess_board,
                                            board_size=board_size, depth=depth - 1, alpha=alpha, beta=beta, direction=d)
            else:
                sim_score = self.alpha_beta(True, my_pos=adv_pos, adv_pos=new_pos, chess_board=chess_board,
                                            board_size=board_size, depth=depth - 1, alpha=alpha, beta=beta, direction=d)

            # undo
            self.undo_barrier(r, c, d, chess_board)

            # possible optimal move

            sim_score["move"] = mv
            #print("after", sim_score, alpha, beta, isMaximizing)

            if isMaximizing:
                if sim_score["score"] > best["score"]:
                    best = sim_score
                    alpha = max(alpha, sim_score["score"])

            else:
                if sim_score["score"] < best["score"]:
                    best = sim_score
                    beta = min(beta, sim_score["score"])

            # Alpha Beta Pruning
            if beta <= alpha:
                break

        #print("End Alpha beta", best, alpha, beta, isMaximizing)
        return best

    def selectMove(self, chess_board, move, adv_pos):

        if len(move) > 10:
            maxMv = 10
        else:
            maxMv = len(move)

        finalList = [(move[0], 0)] * maxMv

        for i in range(maxMv):
            for mv in move:
                r, c, d = mv
                h = self.mvFilter(chess_board, (r, c), adv_pos, d)
                tmp, tmpH = finalList[i]
                if h > tmpH:
                    finalList[i] = ((r, c, d), h)

        for i in range(len(finalList)):
            mv, h = finalList[i]
            finalList[i] = mv

        return finalList





    def mvFilter(self, chess_board, my_pos, adv_pos, direction):
        # calculate the number of walls reachable by both players

        # add more value on direction score if horz or vert wall is needed more

        min_count = 0

        r1, c1 = my_pos
        r2, c2 = adv_pos

        # calculate the direction
        horz = c2 - c1
        vert = r1 - r2

        dirScore = 0

        if horz > 0:
            if direction == 3:
                dirScore -= 2
            elif direction == 1:
                dirScore += 1
        else:
            if direction == 3:
                dirScore += 1
            elif direction == 1:
                dirScore -= 2

        if vert > 0:
            if direction == 2:
                dirScore -= 2
            elif direction == 1:
                dirScore += 1

        else:
            if direction == 2:
                dirScore += 1
            elif direction == 0:
                dirScore -= 2

        trap_scoreM = 0
        trap_scoreA = 0

        for i in range(4):
            if chess_board[r1, c1, i]:
                trap_scoreM -= 6
            if chess_board[r2, c2, i]:
                trap_scoreM += 10

        dis = len(chess_board) / AlphaBetaAgent.manhattan_distance(my_pos, adv_pos)

        # count number of walls around adv and mypos
        # checks if we are in a corner
        for i in range(4):
            if chess_board[r1, c1, i]:
                min_count -= 8
                if r1 % len(chess_board - 1) == 0 and c1 % len(chess_board - 1) == 0:
                    min_count -= 15 * dis
                elif r1 % len(chess_board - 1 == 0):
                    min_count -= 8 * dis
                elif c1 % len(chess_board - 1 == 0):
                    min_count -= 8 * dis
            if chess_board[r2, c2, i]:
                min_count += 5

        new_state = deepcopy(chess_board)

        end_game, p0, p1 = self.check_endgame(len(chess_board), new_state, my_pos, adv_pos)

        count = min_count + dis + dirScore

        # if end_game:
        #     count += (p0 - p1) * 100

        # print("heuristic", count)

        return count

    @staticmethod
    def manhattan_distance(point1, point2):
        distance = 0
        for x1, x2 in zip(point1, point2):
            difference = x2 - x1
            absolute_difference = abs(difference)
            distance += absolute_difference

        return distance

    @staticmethod
    def heuristic(chess_board, move_lista, move_listb, my_pos, adv_pos, direction, isMax):
        # estimates how good a move is
        # move should decrease number of cells opponent can reach while
        # limiting effects on

        # calculate the number of walls reachable by both players

        # add more value on direction score if horz or vert wall is needed more

        max_count = 0
        min_count = 0

        if isMax:
            r1, c1 = my_pos
            r2, c2 = adv_pos
        else:
            r1, c1 = adv_pos
            r2, c2 = my_pos

        # calculate the direction
        horz = c2 - c1
        vert = r1 - r2

        dirScore = 0

        if horz > 0:
            if direction == 3:
                dirScore -= 2
            elif direction == 1:
                dirScore += 1
        else:
            if direction == 3:
                dirScore += 1
            elif direction == 1:
                dirScore -= 2

        if vert > 0:
            if direction == 2:
                dirScore -= 2
            elif direction == 1:
                dirScore += 1

        else:
            if direction == 2:
                dirScore += 1
            elif direction == 0:
                dirScore -= 2

        dis = len(chess_board) / AlphaBetaAgent.manhattan_distance(my_pos, adv_pos)

        # check how many walls surround enemy
        if isMax:
            for r, c, d in move_listb:
                if chess_board[r, c, d]:
                    max_count += 1
        else:
            for r, c, d in move_lista:
                if chess_board[r, c, d]:
                    max_count += 1

        # count number of walls around adv and mypos
        # checks if we are in a corner
        for i in range(4):
            if chess_board[r1, c1, i]:
                min_count -= 12 * dis
                #check if in corner
                if r1 % len(chess_board - 1) == 0 and c1 % len(chess_board - 1) == 0:
                    min_count -= 15 * dis
                elif r1 % len(chess_board - 1 == 0):
                    min_count -= 5 * dis
                elif c1 % len(chess_board - 1 == 0):
                    min_count -= 5
            if chess_board[r2, c2, i]:
                min_count += 5

        # counts number of cell reached omitting direction
        lengthb = 1
        if len(move_listb) > 0:
            seen = move_listb[0]
            for i in move_listb:
                x1, y1, d1, = seen
                x2, y2, d2 = i
                if x1 == x2 and y1 == y2:
                    lengthb += 1
                    seen = i

        lengtha = 1
        if len(move_lista) > 0:
            seen = move_lista[0]
            for i in move_lista:
                x1, y1, d1, = seen
                x2, y2, d2 = i
                if x1 == x2 and y1 == y2:
                    lengtha += 1
                    seen = i

        length = 0

        if isMax:
            if lengthb > lengtha:
                length += (10 / lengthb) - 5
            else:
                length += (10 / lengthb) + 5
        else:
            if lengthb > lengtha:
                length += (10 / lengthb) - 5
            else:
                length += (10 / lengthb) + 5

        count = min_count + dis + dirScore + length + max_count

        #print("heuristic", count)

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
