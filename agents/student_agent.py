# Monte Carlo Tree Search Agent

# System
import sys

# Deepcopy
from copy import deepcopy

# Numpy
import numpy as np

# Default dictionary
from collections import defaultdict

# Agent class
from agents.agent import Agent

# Register agent
from store import register_agent

# Random library
import random

# Math
import math

# Time
import time

# Global Time Limit Variable
timeLimit = 30
currentBoard = 0


# Class to represent the nodes in the tree
class MCTSNode:
    #
    CurrMaxScore = None

    #
    CurrMove = None

    # Initialize the node
    def __init__(self, state, my_pos, adv_pos, is_max, move=None, parent=None, heurist_score=0):

        #
        self.autoplay = True

        #
        self.parent = parent

        #
        self.is_max = is_max

        #
        self.totalScore = 0

        #
        self.numVisit = 0

        #
        self.my_pos = my_pos

        #
        self.adv_pos = adv_pos

        #
        self.move = move

        # Chess_board
        self.state = state

        #
        self.children = []

        #
        self.end_game = False

        #
        self.heuristic_score = heurist_score

    #
    def __str__(self):
        return f"My_pos: {self.my_pos}, Adv_pos: {self.adv_pos}, Score: {self.totalScore}, Visit: {self.numVisit} " \
               f"Move: {self.move}, IsMax: {self.is_max}"

    #
    def __gt__(self, other):
        return self.heuristic_score > other.heuristic_score

    #
    def isTerminal(self):
        return len(self.children) == 0

    #
    def isNewSim(self):
        return self.numVisit == 0

    #
    def getWinRatio(self):
        return self.totalScore / self.numVisit

    #
    def addChild(self, node):
        self.children.append(node)

    #   
    def ucbScore(self):

        num = self.totalScore
        den = self.numVisit
        parentVisit = self.parent.numVisit

        # calculates the UCB
        if parentVisit > 0 and den > 0:
            top = math.log(parentVisit)
            total = num / den + 2 * math.sqrt(top / den)
        else:
            total = math.inf

        return total

    # Selects the child with the best UCB
    def selectBestUcb(self):

        maxUcb = -math.inf
        maxNode = self

        divider = 1
        if len(self.children) > 130:
            divider = 3
        elif len(self.children) > 75:
            divider = 2

        for i in range(len(self.children) // divider):
            child = self.children[i]
            currentUcb = child.ucbScore()
            # Updates node with highest UCB
            if currentUcb > maxUcb:
                maxUcb = currentUcb
                maxNode = child

        return maxNode

    # Selects a child to explore or exploit
    def selectChild(self):
        if self.isTerminal():
            return self

        self.children.sort()
        leafNode = self.selectBestUcb()

        while len(leafNode.children) > 0:
            leafNode.children.sort()
            leafNode = leafNode.selectBestUcb()

        return leafNode


#
@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.autoplay = True
        self.name = "StudentAgent"

        # Moves (Up, Right, Down, Left)
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        # Opposite Directions
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}

        self.root = None

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

        print("\n")
        board_size = len(chess_board)

        # initialize Tree nodes
        self.root = MCTSNode(chess_board, my_pos, adv_pos, False)
        move = self.mcts(chess_board, self.root, max_step, board_size, 50)
        r, x, d = move

        global currentBoard, timeLimit

        if board_size == currentBoard:
            timeLimit = 2
        else:
            timeLimit = 30
            currentBoard = board_size

        print("Board size: ", len(chess_board))
        print("Root Score: ", self.root.totalScore, "Visit", self.root.numVisit, "Move", self.root.move)

        queue = [self.root]

        # parent = None
        # while len(queue) != 0:
        #     node = queue.pop(0)
        #
        #     # if parent == node.parent:
        #     #     print("Root Score: ", node.totalScore, "Visit", node.numVisit, "Move", node.move)
        #     # else:
        #     #     print("\n")
        #     #     print("Root Score: ", node.totalScore, "Visit", node.numVisit, "Move", node.move)
        #     #     parent = node
        #
        #     for child in node.children:
        #         queue.append(child)

        # for child in self.root.children:
        #    print(child)

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

    def mcts(self, chess_board, root, max_step, board_size, num_sim):

        # Start time
        start = time.time()
        # End time
        end = time.time() + timeLimit
        # Root
        parentNode = root

        # Retrieve list of valid moves
        validMove = self.valid_move(chess_board, parentNode.my_pos, max_step, board_size, parentNode.adv_pos)

        # Expansion
        self.expend(chess_board, parentNode, validMove, parentNode.adv_pos, parentNode.is_max)

        # for _ in range(num_sim):
        while start < end:
            # select a leaf to explore
            leaf = parentNode.selectChild()

            if leaf.numVisit > 0:
                validMove = self.valid_move(chess_board, leaf.my_pos, max_step, board_size, leaf.adv_pos)
                self.expend(chess_board, leaf, validMove, leaf.adv_pos, leaf.is_max)

                visit = leaf.selectChild()
            else:
                visit = leaf

            results = self.simulation(visit, max_step, board_size)

            self.backProp(visit, results)

            #print("The time is: ", start)
            start = time.time()

        mv = self.bestMove(root)

        return mv

    # adds children to parents
    def expend(self, chess_board, node, move, adv_pos, turn):
        for mv in move:
            x, y, d = mv
            newState = deepcopy(chess_board)
            self.set_barrier(x, y, d, newState)

            # get a heuristic score for the node
            heur = self.heuristic(newState, (x, y), adv_pos, d, node.is_max)

            # if our turn set child node to False else set to True
            if turn:
                tmpNode = MCTSNode(newState, adv_pos, (x, y), False, mv, node, heur)
            else:
                tmpNode = MCTSNode(newState, adv_pos, (x, y), True, mv, node, heur)

            #check if the node is not an end-game state so we don't expand it
            if not node.end_game:
                node.addChild(tmpNode)
            else:
                break

    def simulation(self, topParent, max_step, board_size):
        chess_board = deepcopy(topParent.state)
        turn = topParent.is_max
        my_pos = topParent.my_pos
        adv_pos = topParent.adv_pos

        end_game, p0_score, p1_score = self.check_endgame(board_size, chess_board, my_pos, adv_pos)

        #if it is a endgame state
        if end_game:
            topParent.end_game = True
            if not turn:
                if p0_score > p1_score:
                    return 3
                elif p0_score == p1_score:
                    return 1
                else:
                    return 0
            else:
                if p0_score < p1_score:
                    return 3
                elif p0_score == p1_score:
                    return 1
                else:
                    return 0

        while not end_game:
            #reverse the turns
            if turn:
                turn = False
            else:
                turn = True

            validMove = self.valid_move(chess_board, my_pos, max_step, board_size, adv_pos)

            selectMv = self.selectBstHeuristic(chess_board, adv_pos, turn, validMove)
            r, c, d = selectMv
            self.set_barrier(r, c, d, chess_board)

            my_pos = adv_pos
            adv_pos = (r, c)

            end_game, p0_score, p1_score = self.check_endgame(board_size, chess_board, my_pos, adv_pos)

        # print("Endgame: ", end_game, "P0 Score: ", p0_score, "P1 Score: ", p1_score, "Turn: ", turn)

        #return the score accordingly
        if not turn:
            if p0_score > p1_score:
                return 2
            elif p0_score == p1_score:
                return 1
            else:
                return 0
        else:
            if p0_score < p1_score:
                return 2
            elif p0_score == p1_score:
                return 1
            else:
                return 0

    def backProp(self, selectedNode, score):
        tmp = selectedNode
        turn = selectedNode.is_max
        tracker = 0

        #if it is our turn and we lost only increase the total score for nodes that aren't us or True
        if turn:
            if score == 0:
                score = 1
                tracker = 1
        else:
            if score == 0:
                score = 1
            else:
                tracker = 1

        #update the node score and visits
        while tmp is not None:
            tmp.numVisit += 1
            if tracker % 2 == 0:
                tmp.totalScore += score
            tracker += 1
            tmp = tmp.parent

    def bestMove(self, root):
        maxI = -100000000
        node = None

        for child in root.children:
            if child.numVisit > 0:
                ratio = child.totalScore / child.numVisit
                if ratio > maxI:
                    maxI = ratio
                    node = child

        # print("TotalScore: ", maxI, " Move: ", node.move, "Num visit", node.numVisit,"total score: " ,node.totalScore)

        return node.move

    def selectBstHeuristic(self, chess_board, adv_pos, isMax, moveList):
        x, y, d = moveList[0]
        finalH = self.heuristic(chess_board, (x, y), adv_pos, d, isMax)
        finalMv = moveList[0]

        for mv in moveList:
            x, y, d = mv
            tmpH = self.heuristic(chess_board, (x, y), adv_pos, d, isMax)
            if tmpH > finalH:
                finalH = tmpH
                finalMv = mv

        return finalMv

    def heuristic(self, chess_board, my_pos, adv_pos, direction, turn):
        # calculate the number of walls reachable by both players

        # add more value on direction score if horz or vert wall is needed more

        min_count = 0


        r1, c1 = my_pos
        r2, c2 = adv_pos

        self.set_barrier(r1, c1, direction, chess_board)

        new_state = deepcopy(chess_board)



        end_game, p0, p1 = self.check_endgame(len(new_state), new_state, my_pos, adv_pos)

        if end_game:
            return p0 - p1

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

        dis = (5 / (np.sqrt(pow((r1 - r2), 2) + pow((c1 - c2), 2)))) + trap_scoreM + trap_scoreA

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

        count = min_count + dis + dirScore

        # print("heuristic", count)

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
