# Student agent: Add your own agent here
# Minimax depth limit and heuristic
import sys
from copy import deepcopy
import numpy as np
from collections import defaultdict
from agents.agent import Agent
from store import register_agent
import random
import math


class MCTSNode:
    CurrMaxScore = None
    CurrMove = None
    
    def __init__(self, state, my_pos, adv_pos, is_max, move=None, parent=None):
        self.parent = parent
        self.is_max = is_max
        self.totalScore = 0
        self.numVisit = 0
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.move = move
        self.state = state # chess_board
        self.children = []
        self.hScore = 0
        
    
    def isTerminal(self):
        return len(self.children) == 0
    
    def isNewSim(self):
        return self.numVisit == 0
    
    def getWinRatio(self):
        return self.totalScore / self.numVisit
    
    def addChild(self, node):
        self.children.append(node)
        
    def ucbScore(self):
        num = self.totalScore
        den = self.numVisit
        parentVisit = self.parent.numVisit
        if parentVisit > 0:
            top = math.log(parentVisit)
        else:
            top = 10000
            
       
            
        if den == 0:
            den = 1
        
        total = num/den + 2 * math.sqrt(top/den)
        
        return total

    def selectBestUcb(self):
        maxUcb = -math.inf
        maxNode = None
        for child in self.children:
            currentUcb = child.ucbScore()
            if currentUcb > maxUcb:
                maxUcb = currentUcb
                maxNode = child
        
        return maxNode
    
    def selectChild(self):
        leafNode = self.selectBestUcb()
        
        while len(leafNode.children) > 0:
            leafNode = leafNode.selectBestUcb()
           
        return leafNode
    

@register_agent("mcts_agent")
class MCTSAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(MCTSAgent, self).__init__()
        self.name = "MCTSAgent"

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

        # bfs to find closest pos to adv
        print("\n")
        board_size = len(chess_board)
        
        self.root = MCTSNode(chess_board, my_pos, adv_pos, True)

        move = self.mcts(chess_board, self.root, max_step, board_size, 200)
        r, x, d = move
        
        print("Root Score: ", self.root.totalScore/self.root.numVisit)

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
        parentNode = root
        validMove = self.valid_move(chess_board, parentNode.my_pos, max_step, board_size, parentNode.adv_pos)
        self.expend(chess_board, parentNode, validMove, parentNode.adv_pos, parentNode.is_max)
        
        for _ in range(num_sim):
            
            leaf = parentNode.selectChild()
            
            
            validMove = self.valid_move(chess_board, leaf.my_pos, max_step, board_size, leaf.adv_pos)
            self.expend(chess_board, leaf, validMove, leaf.adv_pos, leaf.is_max)
            
            visit = leaf.selectChild() 
                
            results = self.simulation(visit, max_step, board_size)

            self.backProp(visit, results)
            
            parentNode = leaf
            
        mv = self.bestMove(root)
    
        return mv
           
    
    def expend(self, chess_board, node, move, adv_pos, turn):
        for mv in move:
            x, y, d = mv 
            newState = deepcopy(chess_board)
            self.set_barrier(x, y, d, newState)
            
            
            if turn:
                tmpNode = MCTSNode(newState, adv_pos, (x,y), False, mv, node)
                
            else:
                tmpNode = MCTSNode(newState, adv_pos, (x,y), True, mv, node)
                
            node.addChild(tmpNode)
    
    def simulation(self, topParent, max_step, board_size):
        chess_board = deepcopy(topParent.state) 
        turn = topParent.is_max
        my_pos = topParent.my_pos
        adv_pos = topParent.adv_pos
        
        end_game, p0_score, p1_score = self.check_endgame(board_size, chess_board, my_pos, adv_pos)     

        while not end_game:
            validMove = self.valid_move(chess_board, my_pos, max_step, board_size, adv_pos)
            
            selectMv = self.selectBstHeuristic(chess_board, adv_pos, turn, validMove)
            r, c, d = selectMv
            self.set_barrier(r, c, d, chess_board)
            
            my_pos = adv_pos
            adv_pos = (r, c)
            
            if turn:
                turn = False
            else:
                turn = True
                
            end_game, p0_score, p1_score = self.check_endgame(board_size, chess_board, my_pos, adv_pos)
            
        #print("Endgame: ", end_game, "P0 Score: ", p0_score, "P1 Score: ", p1_score, "Turn: ", turn)
        if turn:
            if p0_score > p1_score:
                return 1
            return 0
        else:
            if p1_score > p0_score:
                return 1
            return 0
           
                   
    
    def backProp(self, selectedNode, score):
        
        tmp = selectedNode
        
        tmp.totalScore += score
        tmp.numVisit += 1
        
        while tmp.parent != None:
            tmp = tmp.parent
            tmp.numVisit += 1
            tmp.totalScore += score
            
    
    def bestMove(self, root):
        maxI = -100000000
        node = None 
        
        for child in root.children:
            if child.numVisit > 0:
                ratio = child.totalScore / child.numVisit 
                if ratio > maxI:
                    maxI = ratio
                    node = child 
        
        print("TotalScore: ", ratio, " Move: ", node.move)

        return node.move
    
    
    def selectBstHeuristic(self, chess_board, adv_pos, isMax, moveList):
        x, y, d = moveList[0]
        finalH = self.heuristic(chess_board, (x, y), adv_pos, isMax, d)
        finalMv = moveList[0]
        
        for mv in moveList:
            x, y, d = mv
            tmpH = self.heuristic(chess_board, (x, y), adv_pos, isMax, d)
            if tmpH > finalH:
                finalH = tmpH
                finalMv = mv
        
        return finalMv

    @staticmethod
    def heuristic(chess_board, my_pos, adv_pos, isMax, direction):
        # calculate the number of walls reachable by both players

        #add more value on direction score if horz or vert wall is needed more

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

        dis = 5 / (np.sqrt(pow((r1 - r2), 2) + pow((c1 - c2), 2)))


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


    
    
    