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


# Class to represent the nodes in the tree
class MCTSNode:

    # Current score of the node
    CurrMaxScore = None

    # Current movement of the node
    CurrMove = None

    # Initialize the node
    def __init__(self, state, my_pos, adv_pos, is_max, move=None, parent=None, heurist_score=0):

        # Allow autoplay
        self.autoplay = True

        # Every node has a parent
        self.parent = parent

        # Is the maximizing agent
        self.is_max = is_max

        # Total score of the node
        self.totalScore = 0

        # Number of visits to the state
        self.numVisit = 0

        # Position of agent
        self.my_pos = my_pos

        # Position of opponent's agent
        self.adv_pos = adv_pos

        # Current movement
        self.move = move

        # Chess_board
        self.state = state

        # List of children states
        self.children = []

        # Boolean to indicate terminal state
        self.end_game = False

        # Score as determined by heuristic
        self.heuristic_score = heurist_score

    # String function for debugging
    def __str__(self):
        return f"My_pos: {self.my_pos}, Adv_pos: {self.adv_pos}, Score: {self.totalScore}, Visit: {self.numVisit} " \
               f"Move: {self.move}, IsMax: {self.is_max}"

    # Determine which node has greater heurisitc score
    def __gt__(self, other):
        return self.heuristic_score > other.heuristic_score

    # Check if the state is the terminal state
    def isTerminal(self):
        return len(self.children) == 0

    # Check if the node has never been visited
    def isNewSim(self):
        return self.numVisit == 0

    # Compute winning ration = score / visits
    def getWinRatio(self):
        return self.totalScore / self.numVisit

    # Add a child to a node
    def addChild(self, node):
        self.children.append(node)

    # Compute UCB score
    def ucbScore(self):

        # Retrieve score
        num = self.totalScore

        # Retrieve visits
        den = self.numVisit
        
        # Parent node visits
        parentVisit = self.parent.numVisit

        # If visited ~ avoid divide by 0 error
        if parentVisit > 0 and den > 0:

            # Log of parent visits
            top = math.log(parentVisit)
            
            # UCB score
            total = num / den + 2 * math.sqrt(top / den)
        
        # If never visited
        else:
            
            # Infinity
            total = math.inf
        
        # Return UCB score
        return total

    # Selects the child with the best UCB
    def selectBestUcb(self):

        # Current maximum
        maxUcb = -math.inf
        
        # Maximum node
        maxNode = self

        # Value to divide by
        divider = 1
        
        # Greater than 130 branching factor
        if len(self.children) > 130:
            divider = 3
        
        # Greater than 75 branching factor
        elif len(self.children) > 75:
            divider = 2

        # Iterate over chosen children
        for i in range(len(self.children) // divider):
            
            # Retrieve child
            child = self.children[i]
            
            # Compute UCB score
            currentUcb = child.ucbScore()
            
            # Updates node with highest UCB
            if currentUcb > maxUcb:
                
                # Update UCB score
                maxUcb = currentUcb
                
                # Update maximum child
                maxNode = child

        # Return the maximum child
        return maxNode

    # Selects a child to explore or exploit
    def selectChild(self):
        
        # If terminal state
        if self.isTerminal():
            
            # return the state
            return self

        # Sort the children
        self.children.sort()
        
        # Retrieve the best node
        leafNode = self.selectBestUcb()

        # While not at a terminal node
        while len(leafNode.children) > 0:
            
            # Sort the children
            leafNode.children.sort()
            
            # Retrieve the best child
            leafNode = leafNode.selectBestUcb()
        
        # return the best child
        return leafNode

# reigster the agent
@register_agent("student_agent")

# Student Agent
class StudentAgent(Agent):

    # Initialize
    def __init__(self):
        
        # Super class
        super(StudentAgent, self).__init__()
        
        # Allow autoplay
        self.autoplay = True
        
        # Name
        self.name = "StudentAgent"

        # Moves (Up, Right, Down, Left)
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        # Opposite Directions
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}

        # Root is initially empty
        self.root = None

    # Step function
    def step(self, chess_board, my_pos, adv_pos, max_step):
        
        # Board size
        board_size = len(chess_board)

        # initialize Tree nodes
        self.root = MCTSNode(chess_board, my_pos, adv_pos, False)
        
        # Retrieve best move according to MCTS
        move = self.mcts(chess_board, self.root, max_step, board_size, 50)
        
        # Row, column, direction
        r, x, d = move
        
        # Change the time-limit after the first run
        global timeLimit
        
        # New time-limit = 2
        timeLimit = 2

        # Return the move and direction of border
        return (r, x), d

    # Compute list of valid moves
    def valid_move(self, chess_board, my_pos, max_step, board_size, adv_pos):
        
        # List of movements
        move_list = []
        
        # Row, columns for agent
        r, c = my_pos
        
        # Row, column for opponent
        r2, c2 = adv_pos
        
        # Iterate over rows
        for i in range(board_size):
            
            # Iterate over columns
            for j in range(board_size):
                
                # Compute new position
                new_pos = i, j
                
                # Retrieve row and column of position
                x, y = new_pos
                
                # Iterate over directions
                for k in range(4):
                    
                    # If valid step
                    if (self.check_valid_step(np.array([r, c]), np.array([x, y]), k, max_step, chess_board,np.array([r2, c2]))):
                        
                        # Add move to list of moves
                        move_list.append((x, y, k))
        
        # return the list of moves
        return move_list

    # Monte Carlo Tree Search Algorithm
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

            # If node visited before
            if leaf.numVisit > 0:
                
                # Compute list of valid moves
                validMove = self.valid_move(chess_board, leaf.my_pos, max_step, board_size, leaf.adv_pos)
                
                # Expaned tree
                self.expend(chess_board, leaf, validMove, leaf.adv_pos, leaf.is_max)
                
                # Return best child
                visit = leaf.selectChild()
            
            # If node not visited before
            else:
                
                # Do not expand
                visit = leaf

            # Simulate
            results = self.simulation(visit, max_step, board_size)

            # Backpropogate
            self.backProp(visit, results)

            # Start timer
            start = time.time()

        # Compute best move
        mv = self.bestMove(root)

        # Return best move
        return mv

    # Expand
    def expend(self, chess_board, node, move, adv_pos, turn):

        # Iterate over moves
        for mv in move:
            
            # Row, column, direction
            x, y, d = mv
            
            # Copy the state
            newState = deepcopy(chess_board)
            
            # Set the barriers
            self.set_barrier(x, y, d, newState)

            # Compute a heuristic score for the node
            heur = self.heuristic(newState, (x, y), adv_pos, d, node.is_max)

            # Agent's turn
            if turn:
                
                # Set child to false
                tmpNode = MCTSNode(newState, adv_pos, (x, y), False, mv, node, heur)
            
            # Not agent's turn
            else:
                
                # Set child to true
                tmpNode = MCTSNode(newState, adv_pos, (x, y), True, mv, node, heur)

            # Not an end game
            if not node.end_game:
                
                # Add node
                node.addChild(tmpNode)
            
            # End game
            else:
                
                # break
                break
    
    # Simulation
    def simulation(self, topParent, max_step, board_size):
        
        # Copy the board
        chess_board = deepcopy(topParent.state)
        
        # Retrieve turn
        turn = topParent.is_max
        
        # Agent's position
        my_pos = topParent.my_pos
        
        # Opponent's position
        adv_pos = topParent.adv_pos

        # Retrieve end game
        end_game, p0_score, p1_score = self.check_endgame(board_size, chess_board, my_pos, adv_pos)

        # End game state
        if end_game:
            
            # Set root node to true
            topParent.end_game = True
            
            # Not agent's turn
            if not turn:
                
                # Agent won
                if p0_score > p1_score:
                    
                    # return reward
                    return 5
                
                # Agent lost or tied
                else:
                    
                    # return reward
                    return 0
            
            # Agent's turn
            else:
                
                # Agent won
                if p0_score < p1_score:
                    
                    # return reward
                    return 5
                
                # Agent lost
                else:
                    
                    # return reward
                    return 0

        # Not reached terminal state
        while not end_game:
            
            # Agent's turn
            if turn:
                
                # reverse turn
                turn = False
            
            # Opponent's turn
            else:
                
                # reverse turn
                turn = True

            # Compute list of valid moves
            validMove = self.valid_move(chess_board, my_pos, max_step, board_size, adv_pos)

            # Select best move according to heurisitc
            selectMv = self.selectBstHeuristic(chess_board, adv_pos, turn, validMove)
            
            # Row, column, direction
            r, c, d = selectMv
            
            # Set the barriers
            self.set_barrier(r, c, d, chess_board)

            # Switch positions
            my_pos = adv_pos
            
            # Set opponent's position
            adv_pos = (r, c)

            # Compute end game
            end_game, p0_score, p1_score = self.check_endgame(board_size, chess_board, my_pos, adv_pos)

        # Opponent's turn
        if not turn:
            
            # Agent won
            if p0_score > p1_score:
                
                # return reward
                return 1
            
            # Opponent won
            else:
                
                # return reward
                return 0
        
        # Agent's turn
        else:
            
            # Agent won
            if p0_score < p1_score:
                
                # return reward
                return 1
            
            # Opponent won
            else:
                
                # return reward
                return 0

    # Backpropogation
    def backProp(self, selectedNode, score):
        
        # Temporary node
        tmp = selectedNode
        
        # Turn
        turn = selectedNode.is_max
        
        # Tracker
        tracker = 0

        # Agent's turn
        if turn:
            
            # Loss or tie
            if score == 0:
                
                # set score to 1
                score = 1
                
                # update tracker
                tracker = 1
        
        # Opponent's turn
        else:
            
            # Loss or tie
            if score == 0:
                
                # set score to 1
                score = 1
            
            # Win
            else:
                
                # update tracker
                tracker = 1

        # Until root node
        while tmp is not None:

            # increment number of visits
            tmp.numVisit += 1
            
            # every second node
            if tracker % 2 == 0:
                
                # increment score
                tmp.totalScore += score
            
            # increment tracker
            tracker += 1
            
            # update temporary node
            tmp = tmp.parent

    # Compute best move
    def bestMove(self, root):
        
        # Maximum
        maxI = -100000000
        
        # Node
        node = None

        # Iterate over root's children
        for child in root.children:
            
            # Already visited
            if child.numVisit > 0:
                
                # Compute ratio of score to visits
                ratio = child.totalScore / child.numVisit
                
                # Ratio greater than max
                if ratio > maxI:
                    
                    # update the maximum
                    maxI = ratio
                    
                    # update node
                    node = child

        # Return the node
        return node.move

    # Select the best node based on heuristic
    def selectBstHeuristic(self, chess_board, adv_pos, isMax, moveList):
        
        # Row, column, direction
        x, y, d = moveList[0]
        
        # Final heuristic
        finalH = self.heuristic(chess_board, (x, y), adv_pos, d, isMax)
        
        # Final move
        finalMv = moveList[0]

        # Iterate over moves
        for mv in moveList:
            
            # Row, column, direction
            x, y, d = mv
            
            # Temporary heursitic score
            tmpH = self.heuristic(chess_board, (x, y), adv_pos, d, isMax)
            
            # Computed heurisitic score greater than final heuristic score
            if tmpH > finalH:
                
                # update final heuristic
                finalH = tmpH
                
                # update final move
                finalMv = mv

        # return the final move
        return finalMv

    # Heuristic
    def heuristic(self, chess_board, my_pos, adv_pos, direction, turn):

        # add more value on direction score if horz or vert wall is needed more
        min_count = 0

        # agent's row, column
        r1, c1 = my_pos
        
        # opponent's row, column
        r2, c2 = adv_pos

        # set the barriers
        self.set_barrier(r1, c1, direction, chess_board)

        # copy the state
        new_state = deepcopy(chess_board)

        # compute end game
        end_game, p0, p1 = self.check_endgame(len(new_state), new_state, my_pos, adv_pos)

        # reached terminal state
        if end_game:
            
            # return difference in score
            return p0 - p1

        # calculate the horizontal direction
        horz = c2 - c1
        
        # calculate the vertical direction
        vert = r1 - r2

        # score for direction
        dirScore = 0

        # if there is a horizontal direction
        if horz > 0:
            
            # left
            if direction == 3:
                
                # decrement direction score
                dirScore -= 2
            
            # right
            elif direction == 1:
                
                # increment direction score
                dirScore += 1
        
        # no horizontal direction
        else:
            
            # left
            if direction == 3:
                
                # increment direction score
                dirScore += 1
            
            # right
            elif direction == 1:
                
                # decrement direction score
                dirScore -= 2

        # if there is a vertical direction
        if vert > 0:
            
            # down
            if direction == 2:
                
                # decrement score
                dirScore -= 2
            
            # right
            elif direction == 1:
                
                #
                dirScore += 1

        # no vertical direction
        else:
            
            # down
            if direction == 2:
                
                # increment score
                dirScore += 1
            
            # up
            elif direction == 0:
                
                # decrement score
                dirScore -= 2

        # trap score
        trap_scoreM = 0
        
        # trap score
        trap_scoreA = 0

        # iterate over possible number of directions
        for i in range(4):
            
            # if there is a barrier
            if chess_board[r1, c1, i]:
                
                # decrement trap score
                trap_scoreM -= 6
            
            # if there is a barrier
            if chess_board[r2, c2, i]:
                
                # increment trap score
                trap_scoreM += 10

        # compute distance
        dis = (5 / (np.sqrt(pow((r1 - r2), 2) + pow((c1 - c2), 2)))) + trap_scoreM + trap_scoreA

        # iterate over number of possible directions
        for i in range(4):
            
            # if there is a barrier
            if chess_board[r1, c1, i]:
                
                # decrement the minimum counter
                min_count -= 8
                
                # if on diagonal
                if r1 % len(chess_board - 1) == 0 and c1 % len(chess_board - 1) == 0:
                    
                    # decrement the minimum counter
                    min_count -= 15 * dis
                
                # 
                elif r1 % len(chess_board - 1 == 0):
                    
                    # decrement minimum counter
                    min_count -= 8 * dis
                
                # 
                elif c1 % len(chess_board - 1 == 0):
                    
                    # decrement minimum counter
                    min_count -= 8 * dis
            
            # if there is a barrier
            if chess_board[r2, c2, i]:
                
                # increment minimum counter
                min_count += 5

        # update count
        count = min_count + dis + dirScore

        # return count
        return count

    # Check if terminal state reached
    def check_endgame(self, board_size, chess_board, my_pos, adv_pos):
 
        # Union-Find dictionary
        father = dict()
        
        # iterate over rows in board
        for r in range(board_size):
            
            # iterate over columns in board
            for c in range(board_size):
                
                # update dictionary
                father[(r, c)] = (r, c)

        # find the position
        def find(pos):
            
            # if position doesn't exist
            if father[pos] != pos:
                
                # find the parent node's position
                father[pos] = find(father[pos])
            
            # return the position of the parent node
            return father[pos]

        # union
        def union(pos1, pos2):
            
            # update the dictionary
            father[pos1] = pos2

        # iterate over the rows
        for r in range(board_size):
            
            # iterate over the columns
            for c in range(board_size):
                
                # Only check down and right
                for dir, move in enumerate(self.moves[1:3]):

                    # if there is a barrier
                    if chess_board[r, c, dir + 1]:
                        
                        # continue
                        continue
                    
                    # find the position from the dictionary
                    pos_a = find((r, c))
                    
                    # findd the next position from the dictionary
                    pos_b = find((r + move[0], c + move[1]))
                    
                    # positions are not equal
                    if pos_a != pos_b:
                        
                        # union the two positions
                        union(pos_a, pos_b)

        # iterate over the rows
        for r in range(board_size):
            
            # iterate over the columns
            for c in range(board_size):
                
                # find the position from the dictionary
                find((r, c))
        
        # find agent's position
        p0_r = find(my_pos)
        
        # find opponent's position
        p1_r = find(adv_pos)
        
        # compute agent's score
        p0_score = list(father.values()).count(p0_r)
        
        # compute opponent's score
        p1_score = list(father.values()).count(p1_r)

        # scores are equal
        if p0_r == p1_r:
            
            # return false
            return False, p0_score, p1_score

        # return true
        return True, p0_score, p1_score

    # compute the valid steps
    def check_valid_step(self, start_pos, end_pos, barrier_dir, max_step, chess_board, adv_pos):

        # End position
        r, c = end_pos

        # if there is a barrier
        if chess_board[r, c, barrier_dir]:
            
            # return false
            return False

        # start position = end position
        if np.array_equal(start_pos, end_pos):
            
            # return true
            return True

        # Breadth-first search
        state_queue = [(start_pos, 0)]
        
        # visited started position
        visited = {tuple(start_pos)}
        
        # reached target
        is_reached = False

        # while there is states to tranverse and target not reached
        while state_queue and not is_reached:
            
            # current position and step
            cur_pos, cur_step = state_queue.pop(0)
            
            # row, column
            r, c = cur_pos

            # reached maximum step
            if cur_step == max_step:
                
                # break
                break
            
            # iterate over direction and move of all moves
            for dir, move in enumerate(self.moves):
                
                # if there is a barrier
                if chess_board[r, c, dir]:
                    
                    # continue
                    continue
                
                # compute next position
                next_pos = cur_pos + move

                # next position = opponent's position
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    
                    # continue
                    continue
                
                # next position = end position
                if np.array_equal(next_pos, end_pos):
                    
                    # reached target position
                    is_reached = True
                    
                    # break
                    break
                
                # add next position to visited nodes
                visited.add(tuple(next_pos))
                
                # add to tree
                state_queue.append((next_pos, cur_step + 1))

        # return if reached target
        return is_reached

    # set barrier
    def set_barrier(self, r, c, dir, chess_board):
        
        # Set the barrier to True
        chess_board[r, c, dir] = True
        
        # Set the opposite barrier to True
        move = self.moves[dir]
        
        # Set barrier
        chess_board[r + move[0], c + move[1], self.opposites[dir]] = True

    # undo the barrier
    def undo_barrier(self, r, c, dir, chess_board):
        
        # Set the barrier to True
        chess_board[r, c, dir] = False
        
        # Set the opposite barrier to True
        move = self.moves[dir]
        
        # Set barrier
        chess_board[r + move[0], c + move[1], self.opposites[dir]] = False