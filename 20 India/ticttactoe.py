import sys
import math
import time
import random

# Constants
EMPTY = 0
ME = 1
OPPONENT = 2
DRAW = 3

# Game state
game_boards = [[EMPTY for _ in range(9)] for _ in range(9)]
macro_board = [EMPTY for _ in range(9)]
last_move_cell_idx = -1

# Win patterns for 3x3 boards
WINNING_LINES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
    [0, 4, 8], [2, 4, 6]              # Diagonals
]

# Strategic positions (center is most valuable)
POSITION_WEIGHTS = [
    1, 0.8, 1,
    0.8, 2, 0.8,
    1, 0.8, 1
]

def check_mini_board_winner(board):
    """Check if a mini-board has a winner"""
    for line in WINNING_LINES:
        if board[line[0]] == board[line[1]] == board[line[2]] and board[line[0]] != EMPTY:
            return board[line[0]]
    
    if EMPTY not in board:
        return DRAW
    
    return EMPTY

def apply_move(g_boards, m_board, row, col, player):
    """Apply a move to the game state and return the next board index"""
    board_idx = (row // 3) * 3 + (col // 3)
    cell_idx = (row % 3) * 3 + (col % 3)
    
    if g_boards[board_idx][cell_idx] == EMPTY:
        g_boards[board_idx][cell_idx] = player
        
        # Check if this move wins the mini-board
        winner = check_mini_board_winner(g_boards[board_idx])
        if winner != EMPTY:
            m_board[board_idx] = winner
            
    return cell_idx

def get_next_valid_moves(g_boards, m_board, next_board_idx):
    """Get all valid moves based on the next board constraint"""
    valid_moves = []
    
    # If the target board is won or drawn, player can choose any open board
    if m_board[next_board_idx] != EMPTY:
        for board_idx in range(9):
            if m_board[board_idx] == EMPTY:  # Board is still playable
                for cell_idx in range(9):
                    if g_boards[board_idx][cell_idx] == EMPTY:
                        row = (board_idx // 3) * 3 + (cell_idx // 3)
                        col = (board_idx % 3) * 3 + (cell_idx % 3)
                        valid_moves.append((row, col, board_idx, cell_idx))
    else:
        # Must play in the specified board
        board_idx = next_board_idx
        for cell_idx in range(9):
            if g_boards[board_idx][cell_idx] == EMPTY:
                row = (board_idx // 3) * 3 + (cell_idx // 3)
                col = (board_idx % 3) * 3 + (cell_idx % 3)
                valid_moves.append((row, col, board_idx, cell_idx))
                
    return valid_moves

def evaluate_miniboard(miniboard, player):
    """Evaluate the state of a single mini-board"""
    score = 0
    opponent = OPPONENT if player == ME else ME
    
    # Check winning lines
    for line in WINNING_LINES:
        my_count = sum(1 for pos in line if miniboard[pos] == player)
        opp_count = sum(1 for pos in line if miniboard[pos] == opponent)
        empty_count = 3 - my_count - opp_count
        
        if my_count == 3:
            score += 1000
        elif my_count == 2 and empty_count == 1:
            score += 10
        elif opp_count == 2 and empty_count == 1:
            score -= 50
        elif my_count == 1 and empty_count == 2:
            score += 1
    
    # Position weights
    for i in range(9):
        if miniboard[i] == player:
            score += POSITION_WEIGHTS[i]
        elif miniboard[i] == opponent:
            score -= POSITION_WEIGHTS[i]
    
    return score

def evaluate_game_state(g_boards, m_board, player):
    """Evaluate the entire game state"""
    score = 0
    opponent = OPPONENT if player == ME else ME
    
    # Macro board evaluation (main game)
    macro_score = evaluate_miniboard(m_board, player)
    score += macro_score * 1000
    
    # If macro board has a clear win/loss, return immediately
    if abs(macro_score) >= 900:
        return score
    
    # Count won boards
    my_boards = m_board.count(player)
    opp_boards = m_board.count(opponent)
    score += (my_boards - opp_boards) * 100
    
    # Evaluate each mini-board
    for board_idx in range(9):
        if m_board[board_idx] == EMPTY:  # Only evaluate unfinished boards
            board_score = evaluate_miniboard(g_boards[board_idx], player)
            score += board_score
    
    # Strategic considerations
    center_board = m_board[4]  # Center board is most important
    if center_board == player:
        score += 50
    elif center_board == opponent:
        score -= 50
    
    # Mobility - number of available moves in the next turn
    next_moves_count = len(get_next_valid_moves(g_boards, m_board, last_move_cell_idx))
    score += next_moves_count * 0.1
    
    return score

def minimax(g_boards, m_board, depth, alpha, beta, maximizing_player, 
            start_time, time_limit, next_board_idx, use_heuristic=True):
    """Minimax with alpha-beta pruning"""
    
    # Time check
    if time.time() - start_time > time_limit:
        return None, 0
    
    # Check terminal states
    macro_winner = check_mini_board_winner(m_board)
    if macro_winner == ME:
        return None, 100000 + depth * 100  # Prefer quicker wins
    elif macro_winner == OPPONENT:
        return None, -100000 - depth * 100
    elif macro_winner == DRAW:
        return None, 0
    
    if depth == 0:
        if use_heuristic:
            return None, evaluate_game_state(g_boards, m_board, ME)
        else:
            return None, 0
    
    valid_moves = get_next_valid_moves(g_boards, m_board, next_board_idx)
    
    if not valid_moves:
        return None, 0
    
    # Move ordering - sort by heuristic value for better pruning
    if use_heuristic and depth > 1:
        scored_moves = []
        for move in valid_moves:
            row, col, board_idx, cell_idx = move
            temp_boards = [list(b) for b in g_boards]
            temp_macro = list(m_board)
            apply_move(temp_boards, temp_macro, row, col, ME if maximizing_player else OPPONENT)
            score = evaluate_game_state(temp_boards, temp_macro, ME)
            scored_moves.append((move, score))
        
        # Sort by score (high to low for maximizing, low to high for minimizing)
        scored_moves.sort(key=lambda x: x[1], reverse=maximizing_player)
        valid_moves = [move for move, score in scored_moves]
    
    best_move = valid_moves[0]
    
    if maximizing_player:
        max_eval = -math.inf
        for move in valid_moves:
            row, col, board_idx, cell_idx = move
            
            # Create copies for simulation
            temp_boards = [list(b) for b in g_boards]
            temp_macro = list(m_board)
            
            # Apply move
            next_cell = apply_move(temp_boards, temp_macro, row, col, ME)
            
            # Recursive call
            _, eval_score = minimax(temp_boards, temp_macro, depth-1, alpha, beta, 
                                  False, start_time, time_limit, next_cell)
            
            if eval_score is None:  # Timeout
                return best_move, max_eval
                
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
                
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
                
        return best_move, max_eval
    else:
        min_eval = math.inf
        for move in valid_moves:
            row, col, board_idx, cell_idx = move
            
            # Create copies for simulation
            temp_boards = [list(b) for b in g_boards]
            temp_macro = list(m_board)
            
            # Apply move
            next_cell = apply_move(temp_boards, temp_macro, row, col, OPPONENT)
            
            # Recursive call
            _, eval_score = minimax(temp_boards, temp_macro, depth-1, alpha, beta, 
                                  True, start_time, time_limit, next_cell)
            
            if eval_score is None:  # Timeout
                return best_move, min_eval
                
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
                
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
                
        return best_move, min_eval

def get_urgent_moves(g_boards, m_board, next_board_idx, player):
    """Find urgent moves - blocks and immediate wins"""
    urgent_moves = []
    opponent = OPPONENT if player == ME else ME
    
    valid_moves = get_next_valid_moves(g_boards, m_board, next_board_idx)
    
    for move in valid_moves:
        row, col, board_idx, cell_idx = move
        
        # Check if this move wins a mini-board
        temp_boards = [list(b) for b in g_boards]
        temp_macro = list(m_board)
        apply_move(temp_boards, temp_macro, row, col, player)
        
        # Check if we win the mini-board with this move
        if check_mini_board_winner(temp_boards[board_idx]) == player:
            urgent_moves.append((move, 1000))  # High priority
        
        # Check if opponent can win next move in this mini-board
        temp_boards2 = [list(b) for b in g_boards]
        for test_cell in range(9):
            if temp_boards2[board_idx][test_cell] == EMPTY:
                temp_boards2[board_idx][test_cell] = opponent
                if check_mini_board_winner(temp_boards2[board_idx]) == opponent:
                    # Blocking this is important
                    urgent_moves.append((move, 500))
                temp_boards2[board_idx][test_cell] = EMPTY
    
    return [move for move, score in urgent_moves]

# Time management
TIME_LIMIT_FIRST_MOVE = 0.9
TIME_LIMIT_MOVE = 0.095
MAX_DEPTH = 10

# Game loop
first_move = True

while True:
    turn_start = time.time()
    
    # Read inputs
    opponent_row, opponent_col = [int(i) for i in input().split()]
    valid_action_count = int(input())
    
    valid_moves = []
    for i in range(valid_action_count):
        row, col = [int(j) for j in input().split()]
        valid_moves.append((row, col))
    
    # Read board state (we'll use our internal representation primarily)
    for i in range(9):
        input_line = input()  # We'll keep this for compatibility
    
    # Update game state with opponent's move
    if opponent_row != -1:
        first_move = False
        last_move_cell_idx = apply_move(game_boards, macro_board, opponent_row, opponent_col, OPPONENT)
    else:
        first_move = True
        last_move_cell_idx = -1
    
    # Determine time limit for this move
    time_limit = TIME_LIMIT_FIRST_MOVE if first_move else TIME_LIMIT_MOVE
    
    # Check for urgent moves (immediate wins/blocks)
    urgent_moves = get_urgent_moves(game_boards, macro_board, last_move_cell_idx, ME)
    if urgent_moves:
        # Use the first urgent move that's valid
        for urgent_move in urgent_moves:
            row, col, _, _ = urgent_move
            if (row, col) in valid_moves:
                best_move = (row, col)
                break
        else:
            best_move = valid_moves[0]
    else:
        # Use minimax with iterative deepening
        best_move = valid_moves[0]
        best_score = -math.inf
        
        for depth in range(1, MAX_DEPTH + 1):
            if time.time() - turn_start > time_limit * 0.8:
                break
                
            temp_boards = [list(b) for b in game_boards]
            temp_macro = list(macro_board)
            
            move, score = minimax(temp_boards, temp_macro, depth, -math.inf, math.inf, 
                                True, turn_start, time_limit * 0.8, last_move_cell_idx)
            
            if move is not None and score > best_score:
                row, col, _, _ = move
                if (row, col) in valid_moves:
                    best_move = (row, col)
                    best_score = score
            
            # If we found a winning move, use it immediately
            if score > 50000:
                break
    
    # Apply our move to our internal state
    row, col = best_move
    last_move_cell_idx = apply_move(game_boards, macro_board, row, col, ME)
    
    # Output the move
    print(f"{row} {col}")