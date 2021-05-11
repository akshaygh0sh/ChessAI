import time
import numpy as np

# Chess Board:
# Black side
# ---------------
# R N B Q K B N R
# P P P P P P P P


# P P P P P P P P
# R N B Q K B N R
# ---------------
# White side


"""
Static evaluation of the board, returns an floating point number representing the evaluation of the board (0-inf)
does not take into account mate or future captures/lines, that is accounted for via the reinforcement learning training; 
simply factors in doubled pawns, king safety, desirability of squares, etc depending on the stage of the game
(opening, midgame, endgame)
"""

# Alpha Zero piece values
QUEEN_VAL = 950
ROOK_VAL = 563
BISHOP_VAL = 330
KNIGHT_VAL = 305
PAWN_VAL = 100

WHITE_BISHOP_MG = np.array([
    [ -30,   5,  -80,  -35,  -25,  -40,    5,  -10],
    [ -25,  15,  -20,  -15,   30,   60,   20,  -45],
    [ -15,  35,   45,   40,   35,   50,   35,    0],
    [  -5,   5,   20,   50,   35,   35,    5,    0],
    [  -5,  15,   15,   25,   35,   10,   10,    5],
    [   0,  15,   15,   15,   15,   25,   20,   10],
    [   5,  15,   15,    0,    5,   20,   35,    0],
    [ -35,  -5,  -15,  -20,  -15,  -15,  -40,  -20],
])

BLACK_BISHOP_MG = np.flip(WHITE_BISHOP_MG, axis = 0)

WHITE_BISHOP_EG = np.array([
    [ -15,  -20,  -10,  -10,   -5,  -10,  -15,  -25],
    [ -10,   -5,    5,  -10,   -5,  -15,   -5,  -15],
    [   0,  -10,    0,    0,    0,    5,    0,    5],
    [  -5,   10,   10,   10,   15,   10,    5,    0],
    [  -5,    5,   15,   20,   10,   10,   -5,  -10],
    [ -10,   -5,   10,   10,   15,    5,   -5,  -15],
    [ -15,  -20,   -5,    0,    5,  -10,  -15,  -30],
    [ -25,  -10,  -25,   -5,  -10,  -15,   -5,  -15],
])

BLACK_BISHOP_EG = np.flip(WHITE_BISHOP_EG, axis = 0)

WHITE_KNIGHT_MG = np.array([
    [ -165,  -90,  -35,  -50,   60,  -95, -15, -105],
    [  -75,  -40,   70,   35,   25,   60,   5,  -15],
    [  -45,   60,   35,   65,   85,  130,  75,   45],
    [  -10,   15,   20,   55,   35,   70,  20,   20],
    [  -15,    5,   15,   15,   30,   20,  20,  -10],
    [  -25,  -10,   10,   10,   20,   15,  25,  -15],
    [  -30,  -55,  -10,   -5,   0,    20, -15,  -20],
    [ -105,  -20,  -60,  -35,  -15,  -30, -20,  -25]
])

BLACK_KNIGHT_MG = np.flip(WHITE_KNIGHT_MG, axis = 0)

WHITE_KNIGHT_EG = np.array([
    [ -60,  -40,  -15,  -30,  -30,  -25,  -65, -100],
    [ -25,  -10,  -25,    0,  -10,  -25,  -25,  -50],
    [ -25,  -20,   10,   10,    0,  -10,  -20,  -40],
    [ -15,    5,   20,   20,   20,   10,   10,  -20],
    [ -20,   -5,   15,   25,   15,   15,    5,  -20],
    [ -25,   -5,    0,   15,   10,   -5,  -20,  -20],
    [ -40,  -20,  -10,   -5,    0,  -20,  -25,  -45],
    [ -30,  -50,  -25,  -15,  -20,  -20,  -50,  -65]
])

BLACK_KNIGHT_EG = np.flip(WHITE_KNIGHT_EG, axis = 0)

WHITE_ROOK_MG = np.array([
    [ 30,   40,   30,   50,   65,   10,   30,   45],
    [ 30,   30,   60,   60,   80,   65,   25,   45],
    [ -5,   20,   25,   35,   15,   45,   60,   15],
    [-25,  -10,    5,   25,   25,   35,  -10,  -20],
    [-35,  -25,  -10,    0,   10,   -5,    5,  -25],
    [-45,  -25,  -15,  -15,    5,    0,   -5,  -35],
    [-45,  -15,  -20,  -10,    0,   10,   -5,  -70],
    [-20,  -15,    0,   15,   15,    5,  -35,  -25]
])

BLACK_ROOK_MG = np.flip(WHITE_ROOK_MG, axis = 0)

WHITE_ROOK_EG = np.array([
    [ 15,  10,  20,  15,  10,   10,   10,   5],
    [ 10,  15,  15,  10,  -5,    5,   10,   5],
    [  5,   5,   5,   5,   5,   -5,   -5,  -5],
    [  5,   5,  15,   0,   0,    0,    0,   0],
    [  5,   5,  10,   5,  -5,   -5,  -10, -10],
    [ -5,   0,  -5,   0,  -5,  -10,  -10, -15],
    [ -5,  -5,   0,   5, -10,  -10,  -10,  -5],
    [-10,   0,   5,   0,  -5,  -15,    5, -20]
])

BLACK_ROOK_EG = np.flip(WHITE_ROOK_EG, axis = 0)

WHITE_PAWN_MG = np.array([
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [100, 135,  60,  95,  70, 125,  35, -10],
    [ -5,  10,  25,  30,  70,  55,  25, -20],
    [-15,  15,   5,  20,  25,  10,  15, -25],
    [-25,   0,  -5,  10,  15,   5,  10, -25],
    [-25,  -5,  -5, -10,   5,   5,  35, -10],
    [-35,   0, -20, -25, -15,  25,  40, -20],
    [  0,   0,   0,   0,   0,   0,   0,   0]
])

BLACK_PAWN_MG = np.flip(WHITE_PAWN_MG, axis = 0)

WHITE_PAWN_EG = np.array([
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [180, 170, 160, 145, 145, 160, 170, 190],
    [ 10,  10,  20,  30,  30,  20,  10,  10],
    [  5,   5,  10,  25,  25,  10,   5,   5],
    [  0,   0,   0,  20,  20,   0,   0,   0],
    [  5,  -5, -10,   0,   0, -10,  -5,   5],
    [ 15,  10,  10, -20, -20,  10,  10,   5],
    [  0,   0,   0,   0,   0,   0,   0,   0]
])

BLACK_PAWN_EG = np.flip(WHITE_PAWN_EG, axis = 0)

WHITE_QUEEN_MG = np.array([
    [ -30,    0,   30,   10,   60,   45,   45,   45],
    [ -25,  -40,   -5,    0,  -15,   55,   30,   55],
    [ -15,  -15,   10,   10,   30,   55,   45,   55],
    [ -30,  -30,  -15,  -15,    0,   15,    0,    0],
    [ -10,  -25,  -10,  -10,    0,   -5,    5,   -5],
    [ -15,    0,  -10,    0,   -5,    0,   15,    5],
    [ -35,  -10,   10,    0,   10,   15,   -5,    0],
    [   0,  -20,  -10,   10,  -15,  -25,  -30,  -50]
])

BLACK_QUEEN_MG = np.flip(WHITE_QUEEN_MG, axis = 0)

WHITE_QUEEN_EG = np.array([
    [ -10,   20,   20,   25,   25,   20,   10,   20],
    [ -15,   20,   30,   40,   60,   25,   30,    0],
    [ -20,    5,   10,   50,   45,   35,   20,   10],
    [   5,   20,   25,   45,   55,   40,   55,   35],
    [ -20,   30,   20,   45,   30,   35,   40,   25],
    [ -15,  -25,   15,    5,   10,   15,   10,    5],
    [ -20,  -25,  -30,  -15,  -15,  -25,  -35,  -30],
    [ -35,  -30,  -20,  -45,   -5,  -30,  -20,  -40]
])

BLACK_QUEEN_EG = np.flip(WHITE_QUEEN_EG, axis = 0)

WHITE_KING_MG = np.array([
    [ -65,   25,   15,  -15,  -55,  -35,    0,   15],
    [  30,    0,  -20,   -5,  -10,   -5,  -40,  -30],
    [ -10,   25,    0,  -15,  -20,    5,   20,  -20],
    [ -15,  -20,  -10,  -30,  -30,  -25,  -15,  -35],
    [ -50,    0,  -25,  -40,  -45,  -45,  -35,  -50],
    [ -15,  -15,  -20,  -45,  -45,  -30,  -15,  -25],
    [   0,    5,  -10,  -65,  -45,  -15,   10,   10],
    [ -15,   35,   10,  -55,   10,  -30,   25,   15]
])

BLACK_KING_MG = np.flip(WHITE_KING_MG, axis = 0)

WHITE_KING_EG = np.array([
    [ -75,  -35,  -20,  -20,  -10,   15,    5,  -15],
    [ -10,   20,   15,   15,   15,   40,   25,   10],
    [  10,   20,   25,   15,   20,   45,   45,   15],
    [ -10,   20,   25,   25,   25,   35,   25,    5],
    [ -20,   -5,   20,   25,   25,   25,   10,  -10],
    [ -20,   -5,   10,   20,   25,   15,    5,  -10],
    [ -25,  -10,    5,   15,   15,    5,   -5,  -15],
    [ -55,  -35,  -20,  -10,  -30,  -15,  -25,  -45]
])

BLACK_KING_EG = np.flip(WHITE_KING_EG, axis = 0)

def Eval(board, color = True):
    # Index to piece 0 : Pawn, 1 : Knight, 2 : Bishop, 3 : Rook, 4 : Queen, 5: King
    WHITE_PIECE_SQUARES = []
    BLACK_PIECE_SQUARES = []
    if (board.totalMoves < 20):
        WHITE_PIECE_SQUARES = [WHITE_PAWN_MG, WHITE_KNIGHT_MG, WHITE_BISHOP_MG, WHITE_ROOK_MG, WHITE_QUEEN_MG, WHITE_KING_MG]
        BLACK_PIECE_SQUARES = [BLACK_PAWN_MG, BLACK_KNIGHT_MG, BLACK_BISHOP_MG, BLACK_ROOK_MG, BLACK_QUEEN_MG, BLACK_KING_MG]
    else:
        WHITE_PIECE_SQUARES = [WHITE_PAWN_EG, WHITE_KNIGHT_EG, WHITE_BISHOP_EG, WHITE_ROOK_EG, WHITE_QUEEN_EG, WHITE_KING_EG]
        BLACK_PIECE_SQUARES = [BLACK_PAWN_EG, BLACK_KNIGHT_EG, BLACK_BISHOP_EG, BLACK_ROOK_EG, BLACK_QUEEN_EG, BLACK_KING_EG]

    STACKED_PENALTY = 0
    if (board.totalMoves < 15):
        STACKED_PENALTY = 2
    else:
        STACKED_PENALTY = 1.9

    whiteEval = 0
    blackEval = 0

    for col in range(0, 8):
        whitePawnCount = 0
        blackPawnCount = 0
        for row in range(0, 8):
            if (isinstance(board.board[row][col], Queen)):
                if (board.board[row][col].color == True):
                    whiteEval += QUEEN_VAL + WHITE_PIECE_SQUARES[4][row][col]
                else:
                    blackEval += QUEEN_VAL + BLACK_PIECE_SQUARES[4][row][col]
            elif (isinstance(board.board[row][col], Bishop)):
                if (board.board[row][col].color == True):
                    whiteEval += BISHOP_VAL + WHITE_PIECE_SQUARES[2][row][col]
                else:
                    blackEval += BISHOP_VAL + BLACK_PIECE_SQUARES[2][row][col]
            elif (isinstance(board.board[row][col], Rook)):
                if (board.board[row][col].color == True):
                    whiteEval += ROOK_VAL + WHITE_PIECE_SQUARES[3][row][col]
                else:
                    blackEval += ROOK_VAL + BLACK_PIECE_SQUARES[3][row][col]
            elif (isinstance(board.board[row][col], Knight)):
                if (board.board[row][col].color == True):
                    whiteEval += KNIGHT_VAL + WHITE_PIECE_SQUARES[1][row][col]
                else:
                    blackEval += KNIGHT_VAL + BLACK_PIECE_SQUARES[1][row][col]
            elif (isinstance(board.board[row][col], Pawn)):
                if (board.board[row][col].color == True):
                    whiteEval += PAWN_VAL + WHITE_PIECE_SQUARES[0][row][col]
                else:
                    blackEval += PAWN_VAL + BLACK_PIECE_SQUARES[0][row][col]
                # Count the number of stacked pawns
                if (board.board[row][col].color == True):
                    whitePawnCount += 1 
                else:
                    blackPawnCount +=1
            elif (isinstance(board.board[row][col], King)):
                if (board.board[row][col].color == True):
                    whiteEval += WHITE_PIECE_SQUARES[5][row][col]
                else:
                    blackEval += BLACK_PIECE_SQUARES[5][row][col]
        
        pawnCount = 0
        if (color == True):
            pawnCount = whitePawnCount
        else:
            pawnCount = blackPawnCount

        if (pawnCount > 1):
            if (board.totalMoves < 15):
                if (color == True):
                    whiteEval -= (STACKED_PENALTY ** whitePawnCount-1) * 20
                else:
                    blackEval -= (STACKED_PENALTY ** blackPawnCount-1) * 20
            else:
                if (color == True):
                    whiteEval -= (STACKED_PENALTY ** whitePawnCount) * 15
                else:
                    blackEval -= (STACKED_PENALTY ** blackPawnCount) * 15

    # Return white's ratio of evaluation points (whitePoints/total)
    if (color == True):
        return (whiteEval) / (whiteEval + blackEval)
    else:
        return (blackEval) / (whiteEval + blackEval)


# Determines if a square is in Check or not given the board it is placed on and its coordinates
def InCheck(board, coordinates):
    rankNum, fileNum = coordinates

    # Check for pawn checks on White King
    if (isinstance(board[rankNum][fileNum], Piece) and board[rankNum][fileNum].color == True):
        for dx in (-1, 1):
            # Impossible for White King to be in check from a pawn if the white king is on the 7th or 8th ranks
            if (rankNum > 1):
                pawnFile = fileNum + dx
                if (pawnFile >= 0 and pawnFile <= 7):
                    if (isinstance(board[rankNum-1][pawnFile], Pawn) and board[rankNum-1][pawnFile].color != board[rankNum][fileNum].color):
                        return True
            else:
                break
    # Check for pawn checks on Black King
    elif (isinstance(board[rankNum][fileNum], Piece) and board[rankNum][fileNum].color == False):
        for dx in (-1, 1):
            # Impossible for Black King to be in check from a pawn if the black king is on the 1st or 2nd ranks
            if (rankNum < 6):
                pawnFile = fileNum + dx
                if (pawnFile >= 0 and pawnFile <= 7):
                    if (isinstance(board[rankNum+1][pawnFile], Pawn) and board[rankNum+1][pawnFile].color != board[rankNum][fileNum].color):
                        return True
            else:
                break

    # Checks for checks from enemy king (not technically possible, but necessary to generate legal king moves)
    for dy, dx in ((1, 1), (1, -1), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 0), (-1, 1)):
        targetY = rankNum + dy
        targetX = fileNum + dx
        if ((targetX >= 0 and targetX <= 7) and (targetY >= 0 and targetY <= 7)):
            if (isinstance(board[targetY][targetX], King)):
                return True

    # Checks for checks from Knights
    for dy, dx in ((-1, 2), (-1, -2), (1, 2), (1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)):
        targetY = rankNum + dy
        targetX = fileNum + dx
        if ((targetY >= 0 and targetY <= 7) and (targetX >= 0 and targetX <= 7)):
            if (isinstance(board[targetY][targetX], Knight) and board[targetY][targetX].color != board[rankNum][fileNum].color):
                return True

    # Check for vertical checks from rooks/queens
    for y in range(rankNum - 1, -1, -1):
        if (isinstance(board[y][fileNum], Piece) and board[y][fileNum].color != board[rankNum][fileNum].color):
            if (isinstance(board[y][fileNum], Rook)):
                return True
            else:
                break
        elif (isinstance(board[y][fileNum], Piece) and board[y][fileNum].color == board[rankNum][fileNum].color):
            break

    for y in range(rankNum + 1, 8):
        if (isinstance(board[y][fileNum], Piece) and board[y][fileNum].color != board[rankNum][fileNum].color):
            if (isinstance(board[y][fileNum], Rook)):
                return True
            else:
                break
        elif (isinstance(board[y][fileNum], Piece) and board[y][fileNum].color == board[rankNum][fileNum].color):
            break

    # Check for horizontal checks from rooks/queens
    for x in range(fileNum-1, -1, -1):
        if (isinstance(board[rankNum][x], Piece) and board[rankNum][x].color != board[rankNum][fileNum].color):
            if (isinstance(board[rankNum][x], Rook)):
                return True
            else:
                break
        elif (isinstance(board[rankNum][x], Piece) and board[rankNum][x].color == board[rankNum][fileNum].color):
            break

    for x in range(fileNum + 1, 8):
        if (isinstance(board[rankNum][x], Piece) and board[rankNum][x].color != board[rankNum][fileNum].color):
            if (isinstance(board[rankNum][x], Rook)):
                return True
            else:
                break
        elif (isinstance(board[rankNum][x], Piece) and board[rankNum][x].color == board[rankNum][fileNum].color):
            break

    # Check for diagonal checks from bishops/queens
    x = fileNum
    y = rankNum
    # Check north west diagonal checks
    while (x > 0 and y > 0):
        if (board[y-1][x-1] is None):
            y -= 1
            x -= 1
        elif (board[y-1][x-1].color != board[rankNum][fileNum].color):
            if (isinstance(board[y-1][x-1], Bishop)):
                return True
            else:
                break
        else:
            break

    x = fileNum
    y = rankNum
    # Check north east diagonal checks
    while (x < 7 and y > 0):
        if (board[y-1][x+1] is None):
            y -= 1
            x += 1
        elif (board[y-1][x+1].color != board[rankNum][fileNum].color):
            if (isinstance(board[y-1][x+1], Bishop)):
                return True
            else:
                break
        else:
            break

    x = fileNum
    y = rankNum
    # Check south west diagonal checks
    while (x > 0 and y < 7):
        if (board[y+1][x-1] is None):
            y += 1
            x -= 1
        elif (board[y+1][x-1].color != board[rankNum][fileNum].color):
            if (isinstance(board[y+1][x-1], Bishop)):
                return True
            else:
                break
        else:
            break

    x = fileNum
    y = rankNum
    # Check south east diagonal checks
    while (x < 7 and y < 7):
        if (board[y+1][x+1] is None):
            y += 1
            x += 1
        elif (board[y+1][x+1].color != board[rankNum][fileNum].color and isinstance(board[y+1][x+1], Bishop) == True):
            if (isinstance(board[y+1][x+1], Bishop)):
                return True
            else:
                break
        else:
            break

    return False


class Piece:
    legalMoves = set()
    # Color is a bool set to True if the piece is white and False if the piece is black

    def __init__(self, color, rankNum, fileNum, board):
        self.color = color
        self.rankNum = rankNum
        self.fileNum = fileNum
        self.board = board


class Pawn(Piece):
    hasMoved = False

    def MoveList(self):
        self.legalMoves.clear()
        kingCoord = ()
        if (self.color == True):
            kingCoord = self.board.whiteKing
        else:
            kingCoord = self.board.blackKing

        # If pawn hasn't moved
        if (self.hasMoved == False):
            # If pawn is white can move up the board 2 squares
            if (self.color == True):
                if (self.board.board[4][self.fileNum] is None and self.board.board[5][self.fileNum] is None):
                    # Change board as if the potential move was played
                    self.board.board[4][self.fileNum] = self
                    self.board.board[self.rankNum][self.fileNum] = None

                    # Check to see if respective side's king is in check after the move
                    inCheck = InCheck(self.board.board, (kingCoord))
                    if (inCheck != True):
                        self.legalMoves.add(np.ravel_multi_index((self.rankNum, self.fileNum, 1), (8,8,73)))

                    # Change board back to original state
                    self.board.board[4][self.fileNum] = None
                    self.board.board[self.rankNum][self.fileNum] = self
            # If pawn is black can move down the board 2 squares
            else:
                if (self.board.board[3][self.fileNum] is None and self.board.board[2][self.fileNum] is None):
                    # Change board as if the potential move was played
                    self.board.board[3][self.fileNum] = self
                    self.board.board[self.rankNum][self.fileNum] = None

                    # Check to see if respective side's king is in check after the move
                    inCheck = InCheck(self.board.board, (kingCoord))
                    if (inCheck != True):
                        self.legalMoves.add(np.ravel_multi_index((7 - self.rankNum, 7 - self.fileNum, 1), (8,8,73)))

                    # Change board back to original state
                    self.board.board[3][self.fileNum] = None
                    self.board.board[self.rankNum][self.fileNum] = self

        # If pawn is white
        if (self.color == True):
            # If square one in front is empty, pawn can move there
            if (self.board.board[self.rankNum-1][self.fileNum] is None):
                # Change board as if the potential move was played
                self.board.board[self.rankNum-1][self.fileNum] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective side's king is in check after the move
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    if (self.rankNum == 1):
                        # Forward underpromotions
                        for x in [0, 67, 68, 69]:
                            self.legalMoves.add(np.ravel_multi_index((self.rankNum, self.fileNum, x), (8,8,73)))
                    else:
                        self.legalMoves.add(np.ravel_multi_index((self.rankNum, self.fileNum, 0), (8,8,73)))
                
                # Change board back to original state
                self.board.board[self.rankNum-1][self.fileNum] = None
                self.board.board[self.rankNum][self.fileNum] = self

            # If left diagonal pawn capture squares contain black pieces, pawn can move there
            if (self.fileNum - 1 >= 0 and isinstance(self.board.board[self.rankNum-1][self.fileNum-1], Piece) and self.board.board[self.rankNum-1][self.fileNum-1].color != self.color):
                piece = self.board.board[self.rankNum-1][self.fileNum-1]

                # Change board as if the potential move was played
                self.board.board[self.rankNum-1][self.fileNum-1] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective side's king is in check after the move
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    # Left diagonal underpromotions
                    if (self.rankNum == 1):
                        for x in [49, 64, 65, 66]:
                            self.legalMoves.add(np.ravel_multi_index((self.rankNum, self.fileNum, x), (8,8,73)))
                    else:
                        self.legalMoves.add(np.ravel_multi_index((self.rankNum, self.fileNum, 49), (8,8,73)))

                # Change board back to original state
                self.board.board[self.rankNum-1][self.fileNum-1] = piece
                self.board.board[self.rankNum][self.fileNum] = self
            
            # If right diagonal pawn capture square contain black pieces, pawn can move there
            if (self.fileNum + 1 <= 7 and isinstance(self.board.board[self.rankNum-1][self.fileNum+1], Piece) and self.board.board[self.rankNum-1][self.fileNum+1].color != self.color):
                piece = self.board.board[self.rankNum-1][self.fileNum+1]

                # Change board as if the potential move was played
                self.board.board[self.rankNum-1][self.fileNum+1] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective side's king is in check after the move
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    # Right diagonal underpromotions
                    if (self.rankNum == 1):
                        for x in [7, 70, 71, 72]:
                            self.legalMoves.add(np.ravel_multi_index((self.rankNum, self.fileNum, x), (8,8,73)))
                    else:
                        self.legalMoves.add(np.ravel_multi_index((self.rankNum, self.fileNum, 7), (8,8,73)))

                # Change board back to original state
                self.board.board[self.rankNum-1][self.fileNum+1] = piece
                self.board.board[self.rankNum][self.fileNum] = self

            # Check for en passant moves
            if (self.rankNum == 3 and len(self.board.lastMove) == 2 and int(self.board.lastMove[-1]) == 5):
                if (ord(self.board.lastMove[0]) - 97 == self.fileNum-1):
                    # Change board as if potential move was played
                    piece = self.board.board[self.rankNum-1][self.fileNum-1]
                    self.board.board[self.rankNum-1][self.fileNum-1] = self
                    self.board.board[self.rankNum][self.fileNum] = None

                    inCheck = InCheck(self.board.board, (kingCoord))

                    # Change board back to original state
                    self.board.board[self.rankNum-1][self.fileNum-1] = piece
                    self.board.board[self.rankNum][self.fileNum] = self
                    
                    if (inCheck != True):
                        self.legalMoves.add(np.ravel_multi_index((self.rankNum, self.fileNum, 49), (8,8,73)))
                if (ord(self.board.lastMove[0]) - 97 == self.fileNum+1):
                    # Change board as if potential move was played
                    piece = self.board.board[self.rankNum-1][self.fileNum+1]
                    self.board.board[self.rankNum-1][self.fileNum+1] = self
                    self.board.board[self.rankNum][self.fileNum] = None

                    inCheck = InCheck(self.board.board, (kingCoord))

                    # Change board back to original state
                    self.board.board[self.rankNum-1][self.fileNum+1] = piece
                    self.board.board[self.rankNum][self.fileNum] = self
                    
                    if (inCheck != True):
                        self.legalMoves.add(np.ravel_multi_index((self.rankNum, self.fileNum, 7), (8,8,73)))

        # If pawn is black
        else:
            # If square one in front is empty, pawn can move there
            if (self.board.board[self.rankNum+1][self.fileNum] == None):
                # Change board as if the potential move was played
                self.board.board[self.rankNum+1][self.fileNum] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective side's king is in check after the move
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    if (self.rankNum == 6):
                        # Forward underpromotions
                        for x in [0, 67, 68, 69]:
                            self.legalMoves.add(np.ravel_multi_index((7 - self.rankNum, 7 - self.fileNum, x), (8,8,73)))
                    else:
                        self.legalMoves.add(np.ravel_multi_index((7 - self.rankNum, 7 - self.fileNum, 0), (8,8,73)))

                # Change board back to original state
                self.board.board[self.rankNum+1][self.fileNum] = None
                self.board.board[self.rankNum][self.fileNum] = self

            # If right (from black POV) diagonal pawn capture squares contain black pieces, pawn can move there
            if (self.fileNum - 1 >= 0 and isinstance(self.board.board[self.rankNum+1][self.fileNum-1], Piece) and self.board.board[self.rankNum+1][self.fileNum-1].color != self.color):
                piece = self.board.board[self.rankNum+1][self.fileNum-1]

                # Change board as if the potential move was played
                self.board.board[self.rankNum+1][self.fileNum-1] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective side's king is in check after the move
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    if (self.rankNum == 6):
                        for x in [7, 70, 71, 72]:
                            self.legalMoves.add(np.ravel_multi_index((7 - self.rankNum, 7 - self.fileNum, x), (8,8,73)))
                    else:
                        self.legalMoves.add(np.ravel_multi_index((7 - self.rankNum, 7 - self.fileNum, 7), (8,8,73)))

                # Change board back to original state
                self.board.board[self.rankNum+1][self.fileNum-1] = piece
                self.board.board[self.rankNum][self.fileNum] = self

            # If left diagonal pawn capture (from black POV) contains white piece, black pawn can go there
            if (self.fileNum + 1 <= 7 and isinstance(self.board.board[self.rankNum+1][self.fileNum+1], Piece) and self.board.board[self.rankNum+1][self.fileNum+1].color != self.color):
                piece = self.board.board[self.rankNum+1][self.fileNum+1]

                # Change board as if the potential move was played
                self.board.board[self.rankNum+1][self.fileNum+1] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective side's king is in check after the move
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    if (self.rankNum == 6):
                        for x in [49, 64, 65, 66]:
                            self.legalMoves.add(np.ravel_multi_index((7 - self.rankNum, 7 - self.fileNum, x), (8,8,73)))
                    else:
                        self.legalMoves.add(np.ravel_multi_index((7- self.rankNum, 7 - self.fileNum, 49), (8,8,73)))

                # Change board back to original state
                self.board.board[self.rankNum+1][self.fileNum+1] = piece
                self.board.board[self.rankNum][self.fileNum] = self

                # Check for en passant moves
            if (self.rankNum == 4 and len(self.board.lastMove) == 2 and int(self.board.lastMove[-1]) == 4):
                if (ord(self.board.lastMove[0]) - 97 == self.fileNum-1):
                    # Change board as if potential move was played
                    piece = self.board.board[self.rankNum+1][self.fileNum-1]
                    self.board.board[self.rankNum+1][self.fileNum-1] = self
                    self.board.board[self.rankNum][self.fileNum] = None

                    inCheck = InCheck(self.board.board, (kingCoord))

                    # Change board back to original state
                    self.board.board[self.rankNum+1][self.fileNum-1] = piece
                    self.board.board[self.rankNum][self.fileNum] = self
                    
                    if (inCheck != True):
                        self.legalMoves.add(np.ravel_multi_index((7 - self.rankNum, 7 - self.fileNum, 7), (8,8,73)))
                if (ord(self.board.lastMove[0]) - 97 == self.fileNum+1):
                    # Change board as if potential move was played
                    piece = self.board.board[self.rankNum+1][self.fileNum+1]
                    self.board.board[self.rankNum+1][self.fileNum+1] = self
                    self.board.board[self.rankNum][self.fileNum] = None

                    inCheck = InCheck(self.board.board, (kingCoord))

                    # Change board back to original state
                    self.board.board[self.rankNum+1][self.fileNum+1] = piece
                    self.board.board[self.rankNum][self.fileNum] = self
                    
                    if (inCheck != True):
                        self.legalMoves.add(np.ravel_multi_index((7 - self.rankNum, 7 - self.fileNum, 49), (8,8,73)))
        
        return self.legalMoves


class King(Piece):
    hasMoved = False

    def __init__(self, color, rankNum, fileNum, board):
        self.color = color
        self.rankNum = rankNum
        self.fileNum = fileNum
        self.board = board
        if (self.color == True):
            self.board.whiteKing = (self.rankNum, self.fileNum)
        else:
            self.board.blackKing = (self.rankNum, self.fileNum)

    def MoveList(self):
        self.legalMoves.clear()
        # Alpha Zero move codes
        kingMoves = [(1, 1), (1, -1), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 0), (-1, 1)]
        blackMoveCodes = [49, 7, 0, 14, 42, 21, 28, 35]
        whiteMoveCodes = [21, 35, 28, 42, 14, 49, 0, 7]

        flippedRankNum = self.rankNum
        flippedFileNum = self.fileNum

        if (self.color == False):
            flippedRankNum = 7 - self.rankNum
            flippedFileNum = 7 - self.fileNum

        for x in range (len(kingMoves)):
            targetY = self.rankNum + kingMoves[x][0]
            targetX = self.fileNum + kingMoves[x][1]
            if ((targetX >= 0 and targetX <= 7) and (targetY >= 0 and targetY <= 7)):
                if (isinstance(self.board.board[targetY][targetX], Piece) and self.board.board[targetY][targetX].color == self.color):
                    continue
                else:
                    # Change board as if potential move had been played
                    piece = self.board.board[targetY][targetX]
                    self.board.board[targetY][targetX] = self
                    self.board.board[self.rankNum][self.fileNum] = None

                    # Check to see if the king is in check after the move has been played
                    inCheck = InCheck(self.board.board, (targetY, targetX))
                    if (inCheck != True):
                        if (self.color == True):
                            self.legalMoves.add(np.ravel_multi_index((flippedRankNum, flippedFileNum, whiteMoveCodes[x]), (8,8,73)))
                        else:
                            self.legalMoves.add(np.ravel_multi_index((flippedRankNum, flippedFileNum, blackMoveCodes[x]), (8,8,73)))

                    # Change board to original state
                    self.board.board[targetY][targetX] = piece
                    self.board.board[self.rankNum][self.fileNum] = self
                    

        # Check to see if white king can castle
        if (self.color == True and self.hasMoved == False):
            # Check kingside castle
            if (isinstance(self.board.board[7][7], Rook) and self.board.board[7][7].hasMoved == False):
                if (self.board.board[7][5] is None and self.board.board[7][6] is None):
                    canCastle = True
                    for x in (5, 6):
                        self.board.board[7][x] = self
                        self.board.board[7][4] = None

                        canCastle = not InCheck(self.board.board, (7, x))

                        self.board.board[7][x] = None
                        self.board.board[7][4] = self

                        # If the square is attacked, then king can't castle so break
                        if (canCastle == False):
                            break

                    if (canCastle == True):
                        # King can move 2 squares east from white POV
                        self.legalMoves.add(np.ravel_multi_index((flippedRankNum, flippedFileNum, 15), (8,8,73)))
            # Check queenside castle
            if (isinstance(self.board.board[7][0], Rook) and self.board.board[7][0].hasMoved == False):
                if (self.board.board[7][3] is None and self.board.board[7][2] is None and self.board.board[7][1] is None):
                    canCastle = True
                    for x in (3, 2):
                        self.board.board[7][x] = self
                        self.board.board[7][4] = None

                        canCastle = not InCheck(self.board.board, (7, x))

                        self.board.board[7][x] = None
                        self.board.board[7][4] = self

                        # If the square is attacked, then king can't castle so break
                        if (canCastle == False):
                            break

                    if (canCastle == True):
                        # King can move 2 squares west from white POV
                        self.legalMoves.add(np.ravel_multi_index((flippedRankNum, flippedFileNum, 43), (8,8,73)))

        # Check to see if black king can castle
        elif (self.color == False and self.hasMoved == False):
            # Check kingside castle
            if (isinstance(self.board.board[0][7], Rook) and self.board.board[0][7].hasMoved == False):
                if (self.board.board[0][5] is None and self.board.board[0][6]is None):
                    canCastle = True
                    for x in (5, 6):
                        self.board.board[0][x] = self
                        self.board.board[0][4] = None

                        canCastle = not InCheck(self.board.board, (0, x))

                        self.board.board[0][x] = None
                        self.board.board[0][4] = self

                        # If the square is attacked, then king can't castle so break
                        if (canCastle == False):
                            break

                    if (canCastle == True):
                        # King can move 2 squares west from black POV
                        self.legalMoves.add(np.ravel_multi_index((flippedRankNum, flippedFileNum, 43), (8,8,73)))
            # Check queenside castle
            if (isinstance(self.board.board[0][0], Rook) and self.board.board[0][0].hasMoved == False):
                if (self.board.board[0][3] is None and self.board.board[0][2] is None and self.board.board[0][1] is None):
                    canCastle = True
                    for x in (3, 2):
                        self.board.board[0][x] = self
                        self.board.board[0][4] = None

                        canCastle = not InCheck(self.board.board, (0, x))

                        self.board.board[0][x] = None
                        self.board.board[0][4] = self

                        # If the square is attacked, then king can't castle so break
                        if (canCastle == False):
                            break

                    if (canCastle == True):
                        # King can move 2 squares east from black POV
                        self.legalMoves.add(np.ravel_multi_index((flippedRankNum, flippedFileNum, 15), (8,8,73)))
        
        return self.legalMoves


class Bishop(Piece):
    def MoveList(self):
        self.legalMoves.clear()
        kingCoord = ()

        flippedRankNum = self.rankNum
        flippedFileNum = self.fileNum

        if (self.color == True):
            kingCoord = self.board.whiteKing
        else:
            kingCoord = self.board.blackKing
            flippedRankNum = 7 - self.rankNum
            flippedFileNum = 7 - self.fileNum


        x = self.fileNum
        y = self.rankNum

        # Set Alpha Zero Move codes to appropriate direction
        moveCode = 49

        if (self.color == False):
            moveCode = 21

        # Start by checking the legal diagonal moves in the north west direction (from white POV)
        while (x > 0 and y > 0):
            # If the potential square is empty, piece can move there
            if (self.board.board[y-1][x-1] is None):
                # Change board as if potential move had been played
                self.board.board[y-1][x-1] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective king is in check
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    self.legalMoves.add(np.ravel_multi_index((flippedRankNum, flippedFileNum, moveCode), (8,8,73)))

                # Change board back to original state
                self.board.board[y-1][x-1] = None
                self.board.board[self.rankNum][self.fileNum] = self
                y -= 1
                x -= 1
            # If the potential square has a piece of the opposite color, bishop can take that piece but go no further
            # on the diagonal so add that move to the legalMoves and then break
            elif (self.board.board[y-1][x-1].color != self.color):
                piece = self.board.board[y-1][x-1]
                # Change board as if potential move had been played
                self.board.board[y-1][x-1] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective king is in check
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    self.legalMoves.add(np.ravel_multi_index((flippedRankNum, flippedFileNum, moveCode), (8,8,73)))

                # Change board back to original state
                self.board.board[y-1][x-1] = piece
                self.board.board[self.rankNum][self.fileNum] = self
                break
            # If the potential square has a piece of the same color, bishop can not occupy that square or
            # go further on the diagonal, so break
            elif (self.board.board[y-1][x-1].color == self.color):
                break
            moveCode +=1

        x = self.fileNum
        y = self.rankNum

        # Set Alpha Zero Move codes to appropriate direction
        if (self.color == True):
            moveCode = 7
        else:
            moveCode = 35

        # Check the legal diagonal moves in the nort east direction
        while (x < 7 and y > 0):
            if (self.board.board[y-1][x+1] is None):
                # Change board as if potential move had been played
                self.board.board[y-1][x+1] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective king is in check
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    self.legalMoves.add(np.ravel_multi_index((flippedRankNum, flippedFileNum, moveCode), (8,8,73)))

                # Change board back to original state
                self.board.board[y-1][x+1] = None
                self.board.board[self.rankNum][self.fileNum] = self
                y -= 1
                x += 1
            elif (self.board.board[y-1][x+1].color != self.color):
                piece = self.board.board[y-1][x+1]
                # Change board as if potential move had been played
                self.board.board[y-1][x+1] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective king is in check
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    self.legalMoves.add(np.ravel_multi_index((flippedRankNum, flippedFileNum, moveCode), (8,8,73)))

                # Change board back to original state
                self.board.board[y-1][x+1] = piece
                self.board.board[self.rankNum][self.fileNum] = self
                break
            elif (self.board.board[y-1][x+1].color == self.color):
                break
            moveCode+=1

        x = self.fileNum
        y = self.rankNum

        # Set Alpha Zero Move codes to appropriate direction
        if (self.color == True):
            moveCode = 35
        else:
            moveCode = 7

        # Check the legal diagonal moves in the south west direction
        while (x > 0 and y < 7):
            if (self.board.board[y+1][x-1] is None):
                # Change board as if potential move had been played
                self.board.board[y+1][x-1] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective king is in check
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    self.legalMoves.add(np.ravel_multi_index((flippedRankNum, flippedFileNum, moveCode), (8,8,73)))

                # Change board back to original state
                self.board.board[y+1][x-1] = None
                self.board.board[self.rankNum][self.fileNum] = self
                y += 1
                x -= 1
            elif (self.board.board[y+1][x-1].color != self.color):
                piece = self.board.board[y+1][x-1]
                # Change board as if potential move had been played
                self.board.board[y+1][x-1] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective king is in check
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    self.legalMoves.add(np.ravel_multi_index((flippedRankNum, flippedFileNum, moveCode), (8,8,73)))

                # Change board back to original state
                self.board.board[y+1][x-1] = piece
                self.board.board[self.rankNum][self.fileNum] = self
                break
            elif (self.board.board[y+1][x-1].color == self.color):
                break
            moveCode+=1

        x = self.fileNum
        y = self.rankNum

        # Set Alpha Zero Move codes to appropriate direction
        if (self.color == True):
            moveCode = 21
        else:
            moveCode = 49

        # Check the legal diagonal moves in the south east direction
        while (x < 7 and y < 7):
            if (self.board.board[y+1][x+1] is None):
                # Change board as if potential move had been played
                self.board.board[y+1][x+1] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective king is in check
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    self.legalMoves.add(np.ravel_multi_index((flippedRankNum, flippedFileNum, moveCode), (8,8,73)))

                # Change board back to original state
                self.board.board[y+1][x+1] = None
                self.board.board[self.rankNum][self.fileNum] = self
                y += 1
                x += 1

            elif (self.board.board[y+1][x+1].color != self.color):
                piece = self.board.board[y+1][x+1]
                # Change board as if potential move had been played
                self.board.board[y+1][x+1] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective king is in check
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    self.legalMoves.add(np.ravel_multi_index((flippedRankNum, flippedFileNum, moveCode), (8,8,73)))

                # Change board back to original state
                self.board.board[y+1][x+1] = piece
                self.board.board[self.rankNum][self.fileNum] = self
                break
            elif (self.board.board[y+1][x+1].color == self.color):
                break
            moveCode+=1

        return self.legalMoves


class Knight(Piece):
    def MoveList(self):
        self.legalMoves.clear()
        kingCoord = ()

        knightMoves = [(-1, 2), (-1, -2), (1, 2), (1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)]
        blackMoveCodes = [61, 58, 62, 57, 63, 56, 60, 59]
        whiteMoveCodes = [57, 62, 58, 61, 59, 60, 56, 63]

        flippedRankNum = self.rankNum
        flippedFileNum = self.fileNum

        if (self.color == True):
            kingCoord = self.board.whiteKing
        else:
            kingCoord = self.board.blackKing
            flippedRankNum = 7 - self.rankNum
            flippedFileNum = 7 - self.fileNum

        for x in range (0, len(knightMoves)):
            newRank = self.rankNum + knightMoves[x][0]
            newFile = self.fileNum + knightMoves[x][1]
            # Test that possible knight move is still in the self.board.board.board
            if ((newRank >= 0 and newRank <= 7) and (newFile >= 0 and newFile <= 7)):
                # If the potential knight move is on a empty square or enemy piece, it's legal
                if (self.board.board[newRank][newFile] is None or self.board.board[newRank][newFile].color != self.color):
                    piece = self.board.board[newRank][newFile]
                    self.board.board[newRank][newFile] = self
                    self.board.board[self.rankNum][self.fileNum] = None

                    # Check to see if king is in check after move (i.e. piece isn't pinned)
                    inCheck = InCheck(self.board.board, (kingCoord))
                    if (inCheck != True):
                        if (self.color == True):
                            self.legalMoves.add(np.ravel_multi_index((flippedRankNum, flippedFileNum, whiteMoveCodes[x]), (8,8,73)))
                        else:
                            self.legalMoves.add(np.ravel_multi_index((flippedRankNum, flippedFileNum, blackMoveCodes[x]), (8,8,73)))
                    
                    self.board.board[newRank][newFile] = piece
                    self.board.board[self.rankNum][self.fileNum] = self

        return self.legalMoves


class Rook (Piece):
    hasMoved = False

    def MoveList(self):
        self.legalMoves.clear()
        kingCoord = ()

        # Alpha Zero move encoding val
        moveCode = 0

        flippedRankNum = self.rankNum
        flippedFileNum = self.fileNum

        if (self.color == True):
            kingCoord = self.board.whiteKing
        else:
            kingCoord = self.board.blackKing
            # Adjusted coordinates after "flipping" board
            flippedRankNum = 7 - self.rankNum
            flippedFileNum = 7 - self.fileNum


        # If piece is black, following tests southward moves
        if (self.color == False):
            moveCode = 28

        # Check all legal vertical moves from rook's current position
        for y in range(self.rankNum - 1, -1, -1):
            # If potential position is empty, rook can move there
            if (self.board.board[y][self.fileNum] is None):
                # Change board as if potential move had been played
                self.board.board[y][self.fileNum] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective side king is in check after the move has been played
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    self.legalMoves.add(np.ravel_multi_index((flippedRankNum, flippedFileNum, moveCode), (8,8,73)))

                # Change board back to original state
                self.board.board[y][self.fileNum] = None
                self.board.board[self.rankNum][self.fileNum] = self
            # If potential position has a piece of the opposite color, rook can take that piece but go no further, so add that move
            # to the legalMoves and then break
            elif (self.board.board[y][self.fileNum].color != self.color):
                piece = self.board.board[y][self.fileNum]
                # Change board as if potential move had been played
                self.board.board[y][self.fileNum] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective side king is in check after the move has been played
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    self.legalMoves.add(np.ravel_multi_index((flippedRankNum, flippedFileNum, moveCode), (8,8,73)))

                # Change board back to original state
                self.board.board[y][self.fileNum] = piece
                self.board.board[self.rankNum][self.fileNum] = self
                break
            # If potential position has a piece of the same color, rook can't go there, so break
            elif (self.board.board[y][self.fileNum].color == self.color):
                break
            moveCode+=1

        if (self.color == True):
            moveCode = 28
        else:
            moveCode = 0

        # Check all legal vertical downward moves from rook's current position
        for y in range(self.rankNum + 1, 8):
            if (self.board.board[y][self.fileNum] is None):
                # Change board as if potential move had been played
                self.board.board[y][self.fileNum] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective side king is in check after the move has been played
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    self.legalMoves.add(np.ravel_multi_index((flippedRankNum, flippedFileNum, moveCode), (8,8,73)))

                # Change board back to original state
                self.board.board[y][self.fileNum] = None
                self.board.board[self.rankNum][self.fileNum] = self

            elif (self.board.board[y][self.fileNum].color != self.color):
                piece = self.board.board[y][self.fileNum]
                # Change board as if potential move had been played
                self.board.board[y][self.fileNum] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective side king is in check after the move has been played
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    self.legalMoves.add(np.ravel_multi_index((flippedRankNum, flippedFileNum, moveCode), (8,8,73)))

                # Change board back to original state
                self.board.board[y][self.fileNum] = piece
                self.board.board[self.rankNum][self.fileNum] = self
                break
            elif (self.board.board[y][self.fileNum].color == self.color):
                break
            moveCode+=1
        
        if (self.color == True):
            moveCode = 42
        else:
            moveCode = 14

        # Check all legal horizontal moves from rook's current position
        for x in range(self.fileNum - 1, -1, -1):
            if (self.board.board[self.rankNum][x] is None):
                # Change board as if potential move had been played
                self.board.board[self.rankNum][x] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective side king is in check after the move has been played
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    self.legalMoves.add(np.ravel_multi_index((flippedRankNum, flippedFileNum, moveCode), (8,8,73)))

                # Change board back to original state
                self.board.board[self.rankNum][x] = None
                self.board.board[self.rankNum][self.fileNum] = self

            elif (self.board.board[self.rankNum][x].color != self.color):
                piece = self.board.board[self.rankNum][x]
                # Change board as if potential move had been played
                self.board.board[self.rankNum][x] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective side king is in check after the move has been played
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    self.legalMoves.add(np.ravel_multi_index((flippedRankNum, flippedFileNum, moveCode), (8,8,73)))

                # Change board back to original state
                self.board.board[self.rankNum][x] = piece
                self.board.board[self.rankNum][self.fileNum] = self
                break
            elif (self.board.board[self.rankNum][x].color == self.color):
                break
            moveCode+=1

        if (self.color == True):
            moveCode = 14
        else:
            moveCode = 42
        # Check all legal horizontal moves from the rook's current position to the top of the self.board.board
        for x in range(self.fileNum + 1, 8):
            if (self.board.board[self.rankNum][x] is None):
                # Change board as if potential move had been played
                self.board.board[self.rankNum][x] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective side king is in check after the move has been played
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    self.legalMoves.add(np.ravel_multi_index((flippedRankNum, flippedFileNum, moveCode), (8,8,73)))

                # Change board back to original state
                self.board.board[self.rankNum][x] = None
                self.board.board[self.rankNum][self.fileNum] = self
            elif (self.board.board[self.rankNum][x].color != self.color):
                piece = self.board.board[self.rankNum][x]
                # Change board as if potential move had been played
                self.board.board[self.rankNum][x] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective side king is in check after the move has been played
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    self.legalMoves.add(np.ravel_multi_index((flippedRankNum, flippedFileNum, moveCode), (8,8,73)))

                # Change board back to original state
                self.board.board[self.rankNum][x] = piece
                self.board.board[self.rankNum][self.fileNum] = self
                break
            elif (self.board.board[self.rankNum][x].color == self.color):
                break
            moveCode+=1

        return self.legalMoves


class Queen(Bishop, Rook):
    def MoveList(self):
        # Queen moves are just a combination of the set of possible rook moves and possible bishop moves.
        self.legalMoves = Bishop.MoveList(self).copy() | Rook.MoveList(self).copy()
        return self.legalMoves


class Board ():
    

    def __init__(self):
        board = []
        # Initalize board to 8 by 8
        for x in range (8):
            board.append([None] * 8)
        
        # Set the position of the pieces
        for x in range(0, 8):
            # Set pawn positions
            board[1][x] = Pawn(False, 1, x, self)
            board[6][x] = Pawn(True, 6, x, self)

            # Set rook positions
            if (x == 0 or x == 7):
                board[0][x] = Rook(False, 0, x, self)
                board[7][x] = Rook(True, 7, x, self)

            # Set bishop positions
            if (x == 2 or x == 5):
                board[0][x] = Bishop(False, 0, x, self)
                board[7][x] = Bishop(True, 7, x, self)

            # Set knight positions
            if (x == 1 or x == 6):
                board[0][x] = Knight(False, 0, x, self)
                board[7][x] = Knight(True, 7, x, self)

            # Set queen positions
            if (x == 3):
                board[0][x] = Queen(False, 0, x, self)
                board[7][x] = Queen(True, 7, x, self)

            # Set king positions
            if (x == 4):
                board[0][x] = King(False, 0, x, self)
                self.blackKing = ((0, x))
                board[7][x] = King(True, 7, x, self)
                self.whiteKing = ((7, x))
        self.board = board

        self.listOfMoves = []
        self.lastMove = ""
        self.totalMoves = 1
        self.whiteToMove = True
        self.gameState = 0
        self.lastCapture = 0
        self.allLegalMoves = set ()

        # Move decoder takes move codes and turns them into their respective move tuples
        self.moveDecoder = dict ()
        for moveCode in range (0, 73):
            if (moveCode < 56):
                    direction = moveCode // 7
                    magnitude = moveCode % 7 + 1
                    # North move
                    if (direction == 0):
                        self.moveDecoder[moveCode] = (-magnitude, 0)
                    # Northeast move
                    elif (direction == 1):
                        self.moveDecoder[moveCode] = (-magnitude, magnitude)
                    # East move
                    elif (direction == 2):
                        self.moveDecoder[moveCode] = (0, magnitude)
                    # Southeast move
                    elif (direction == 3):
                        self.moveDecoder[moveCode] = (magnitude, magnitude)
                    # South move
                    elif (direction == 4):
                        self.moveDecoder[moveCode] = (magnitude, 0)
                    # Southwest move
                    elif (direction == 5):
                        self.moveDecoder[moveCode] = (magnitude, -magnitude)
                    # West move
                    elif (direction == 6):
                        self.moveDecoder[moveCode] = (0, -magnitude)
                    # Nortwest move
                    elif (direction == 7):
                        self.moveDecoder[moveCode] = (-magnitude, -magnitude)
            # Knight moves
            elif (moveCode >= 56 and moveCode < 64):
                if (moveCode == 56):
                    self.moveDecoder[moveCode] = (-2, 1)
                elif (moveCode == 57):
                    self.moveDecoder[moveCode] = (-1, 2)
                elif (moveCode == 58):
                    self.moveDecoder[moveCode] = (1, 2)
                elif (moveCode == 59):
                    self.moveDecoder[moveCode] = (2, 1)
                elif (moveCode == 60):
                    self.moveDecoder[moveCode] = (2, -1)
                elif (moveCode == 61):
                    self.moveDecoder[moveCode] = (1, -2)
                elif (moveCode == 62):
                    self.moveDecoder[moveCode] = (-1, -2)
                elif (moveCode == 63):
                    self.moveDecoder[moveCode] = (-2, -1)
            # Under promotions
            elif (moveCode >= 64 and moveCode <= 72): 
                # Left pawn capture from white POV to promote to Knight, Bishop, then Rook respectively
                if (moveCode == 64):
                    self.moveDecoder[moveCode] = (-1, -1)
                elif (moveCode == 65):
                    self.moveDecoder[moveCode] = (-1, -1)
                elif (moveCode == 66):
                    self.moveDecoder[moveCode] = (-1, -1)
                # Forward pawn move from white POV to promote to Knight, Bishop, then Rook respectively
                elif (moveCode == 67):
                    self.moveDecoder[moveCode] = (-1, 0)
                elif (moveCode == 68):
                    self.moveDecoder[moveCode] = (-1, 0)
                elif (moveCode == 69):
                    self.moveDecoder[moveCode] = (-1, 0)
                # Right pawn capture from white POV to promote to Knight, Bishop, then Rook respectively
                elif (moveCode == 70):
                    self.moveDecoder[moveCode] = (-1, 1)
                elif (moveCode == 71):
                    self.moveDecoder[moveCode] = (-1, 1)
                elif (moveCode == 72):
                    self.moveDecoder[moveCode] = (-1, 1)
        # Move encoder takes move tuples and turns them into their respective move codes, moveEncoder doesn't work on moves 64-72
        self.moveEncoder = dict ()
        for (moveCode, moveTuple) in self.moveDecoder.items():
            if (moveTuple not in self.moveEncoder):
                self.moveEncoder[moveTuple] = moveCode

    # Resets the board entirely, including all its data fields
    def InitializeBoard (self):
        self.listOfMoves = []
        self.lastMove = ""
        self.totalMoves = 1
        self.whiteToMove = True
        self.gameState = 0
        self.lastCapture = 0
        for x in range(0, 8):
            # Set pawn positions
            self.board[1][x] = Pawn(False, 1, x, self)
            self.board[6][x] = Pawn(True, 6, x, self)

            # Set rook positions
            if (x == 0 or x == 7):
                self.board[0][x] = Rook(False, 0, x, self)
                self.board[7][x] = Rook(True, 7, x, self)

            # Set bishop positions
            if (x == 2 or x == 5):
                self.board[0][x] = Bishop(False, 0, x, self)
                self.board[7][x] = Bishop(True, 7, x, self)

            # Set knight positions
            if (x == 1 or x == 6):
                self.board[0][x] = Knight(False, 0, x, self)
                self.board[7][x] = Knight(True, 7, x, self)

            # Set queen positions
            if (x == 3):
                self.board[0][x] = Queen(False, 0, x, self)
                self.board[7][x] = Queen(True, 7, x, self)

            # Set king positions
            if (x == 4):
                self.board[0][x] = King(False, 0, x, self)
                self.blackKing = ((0, x))
                self.board[7][x] = King(True, 7, x, self)
                self.whiteKing = ((7, x))
            
        # Set all other squares to None
        for row in range (2, 6):
            for col in range (0, 8):
                self.board[row][col] = None

    # Returns the decoded move tuple of the corresponding move code (from current player POV)
    def DecodeMove (self, moveCode):
        return self.moveDecoder[moveCode]
    
    # Returns the corresponding move code given a move tuple (from current player POV)
    def EncodeMove (self, moveTuple, underpromotion = ""):
        # Mirror the move (-dy, -dx) if it's black's turn
        if (self.whiteToMove == False):
            moveTuple = (-1 * moveTuple[0], -1 * moveTuple[1])

        # If the move tuple isn't in the move encoder, it's illegal so return None/null
        if (moveTuple not in self.moveEncoder):
            return None   

        if (underpromotion == "R"):
            if (moveTuple == (-1, -1)):
                return 66
            elif (moveTuple == (-1, 0)):
                return 69
            elif (moveTuple == (-1, 1)):
                return 72
        elif (underpromotion == "N"):
            if (moveTuple == (-1, -1)):
                return 64
            elif (moveTuple == (-1, 0)):
                return 67
            elif (moveTuple == (-1, 1)):
                return 71
        elif (underpromotion == "B"):
            if (moveTuple == (-1, -1)):
                return 65
            elif (moveTuple == (-1, 0)):
                return 68
            elif (moveTuple == (-1, 1)):
                return 71
        return self.moveEncoder[moveTuple]

    # Returns algebraic notation of a move given the move coordinate (y, x, c) from player to move's POV
    def CoordToAlgebraic(self, coordinate, takes):
        rankNum, fileNum, moveCode = coordinate

        piece = self.board[rankNum][fileNum]
        moveInfo = self.moveDecoder[moveCode]

        prefix = ""
        files = { 0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h" }

        # Mirror the coordinates if it's black's turn
        if (self.whiteToMove == False):
            piece = self.board[7 - rankNum][7 - fileNum]
            moveInfo = (moveInfo[0] * -1, moveInfo[1] * -1)
        else:
            prefix = "".join([str(self.totalMoves), ". "])
        
        if (isinstance(piece, King) == True):
            prefix = "".join([prefix, "K"])
        elif (isinstance(piece, Queen) == True):
            prefix = "".join([prefix, "Q"])
        elif (isinstance(piece, Rook) == True):
            prefix = "".join([prefix, "R"])
        elif (isinstance(piece, Bishop) == True):
            prefix = "".join([prefix, "B"])
        elif (isinstance(piece, Knight) == True):
            prefix = "".join([prefix, "N"])
        elif (isinstance (piece, Pawn) == True and takes == True):
            prefix = "".join([prefix, files[piece.fileNum]])

        # Castling
        if (isinstance(piece, King) == True):
            if (moveCode == 15):
                if (self.whiteToMove == True):
                    return "".join([str(self.totalMoves), ". 0-0 "])
                else:
                    return "0-0-0 "
            elif (moveCode == 43):
                if (self.whiteToMove == True):
                    return "".join([str(self.totalMoves), ". 0-0-0 "])
                else:
                    return "0-0 "

        targetY = piece.rankNum + moveInfo[0]
        targetX = piece.fileNum + moveInfo[1]
        samePiece = False
        sameRow = False
        sameCol = False

        # Disambiguating moves

        # Knights
        if (isinstance(piece, Knight)):
            # Checks for other knights that can go to same square
            for dy, dx in ((-1, 2), (-1, -2), (1, 2), (1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)):
                yCoord = targetY + dy
                xCoord = targetX + dx
                if ((yCoord >= 0 and yCoord <= 7) and (xCoord >= 0 and xCoord <= 7)):
                    if (isinstance(self.board[yCoord][xCoord], Knight) == True and self.board[yCoord][xCoord].color == piece.color and (yCoord, xCoord) != (piece.rankNum, piece.fileNum)):
                        samePiece = True
                        if (yCoord == piece.rankNum):
                            sameRow = True
                        if (xCoord == piece.fileNum):
                            sameCol = True

        # Rooks
        if (isinstance(piece, Rook)):
            # Check for other rooks on same file (up the board from white POV)
            for y in range(targetY - 1, -1, -1):
                if (self.board[y][targetX] is not None):
                    if (isinstance(self.board[y][targetX], Rook) and self.board[y][targetX].color == piece.color and (y, targetX) != (piece.rankNum, piece.fileNum)):
                        if (type(piece) == type(self.board[y][targetX])):
                            samePiece = True
                            if (targetX == piece.fileNum):
                                sameCol = True
                    else:
                        break
            
            # Check for other rooks on same file (down the board from white POV)
            for y in range (targetY + 1, 8):
                if (self.board[y][targetX] is not None):
                    if (isinstance(self.board[y][targetX], Rook) and self.board[y][targetX].color == piece.color and (y, targetX) != (piece.rankNum, piece.fileNum)):
                        if (type(piece) == type(self.board[y][targetX])):
                            samePiece = True
                            if (targetX == piece.fileNum):
                                sameCol = True
                    else:
                        break
            
            # Check for rooks on same rank to left of current piece (from white POV)
            for x in range (targetX -1, -1, -1):
                if (self.board[piece.rankNum][x] is not None):
                    if (isinstance(self.board[targetY][x], Rook) and self.board[targetY][x].color == piece.color and (targetY, x) != (piece.rankNum, piece.fileNum)):
                        if (type(piece) == type (self.board[targetY][x])):
                            samePiece = True
                            if (targetY == piece.rankNum):
                                sameRow = True
                    else:
                        break
            
            # Check for rooks on same rank to the right of current piece (from white POV)
            for x in range (targetX + 1, 8):
                if (self.board[targetY][x] is not None):
                    if (isinstance(self.board[targetY][x], Rook) and self.board[targetY][x].color == piece.color and (targetY, x) != (piece.rankNum, piece.fileNum)):
                        if (type(piece) == type (self.board[targetY][x])):
                            samePiece = True
                            if (targetY == piece.rankNum):
                                sameRow = True
                    else:
                        break
        
        # Bishops
        if (isinstance(piece, Bishop)):
            x = targetX + 1
            y = targetY - 1

            # Check for bishops on the north east diagonal (white POV)
            while (x <= 7 and y >= 0):
                if (self.board[y][x] is not None):
                    if (self.board[y][x].color == piece.color and isinstance(self.board[y][x], Bishop) and (y, x) != (piece.rankNum, piece.fileNum)):
                        if (type(self.board[y][x]) == type(piece)):
                            samePiece = True
                            if (y == piece.rankNum):
                                sameRow = True
                            if (x == piece.fileNum):
                                sameCol = True
                    break
                else:
                    x += 1
                    y -= 1
            
            # Check for bishops on the nort west diagonal (white POV)
            x = targetX - 1
            y = targetY - 1

            # Check for bishops on the north west diagonal (white POV)
            while (x >= 0 and y >= 0):
                if (self.board[y][x] is not None):
                    if (self.board[y][x].color == piece.color and isinstance(self.board[y][x], Bishop) and (y, x) != (piece.rankNum, piece.fileNum)):
                        if (type(self.board[y][x]) == type(piece)):
                            samePiece = True
                            if (y == piece.rankNum):
                                sameRow = True
                            if (x == piece.fileNum):
                                sameCol = True
                    break
                else:
                    x -= 1
                    y -= 1
            
            x = targetX + 1
            y = targetY + 1

            # Check for bishops on the south east diagonal (white POV)
            while (x <= 7 and y <= 7):
                if (self.board[y][x] is not None):
                    if (self.board[y][x].color == piece.color and isinstance(self.board[y][x], Bishop) and (y, x) != (piece.rankNum, piece.fileNum)):
                        if (type(self.board[y][x]) == type(piece)):
                            samePiece = True
                            if (y == piece.rankNum):
                                sameRow = True
                            if (x == piece.fileNum):
                                sameCol = True
                    break
                else:
                    x += 1
                    y += 1

            x = targetX - 1
            y = targetY + 1

            # Check for bishops on the south west diagonal (white POV)
            while (x >= 0 and y <= 7):
                if (self.board[y][x] is not None):
                    if (self.board[y][x].color == piece.color and isinstance(self.board[y][x], Bishop) and (y, x) != (piece.rankNum, piece.fileNum)):
                        if (type(self.board[y][x]) == type(piece)):
                            samePiece = True
                            if (y == piece.rankNum):
                                sameRow = True
                            if (x == piece.fileNum):
                                sameCol = True
                    break
                else:
                    x -= 1
                    y += 1

        if (samePiece == True):
            if (sameCol == True and sameRow == True):
                prefix = "".join([prefix, files[piece.fileNum], str(8 - piece.rankNum)])
            elif (sameCol == True):
                prefix = "".join([prefix, str(8 - piece.rankNum)])
            else:
                prefix = "".join([prefix, files[piece.fileNum]])

        if (takes == True):
            prefix = "".join([prefix, "x"])

        if (isinstance(piece, Pawn)):
            if ((piece.color == True and piece.rankNum == 1) or (piece.color == False and piece.rankNum == 6)):
                if (moveCode == 0 or moveCode == 7 or moveCode == 49):
                    return "".join([prefix, files[targetX], str(8 - targetY), "Q"])
                elif (moveCode == 64 or moveCode == 67 or moveCode == 70):
                    return "".join([prefix, files[targetX], str(8 - targetY), "N"])
                elif (moveCode == 65 or moveCode == 68 or moveCode == 71):
                    return "".join([prefix, files[targetX], str(8 - targetY), "B"])
                elif (moveCode == 66 or moveCode == 69 or moveCode == 72):
                    return "".join([prefix, files[targetX], str(8 - targetY), "R"])
                
        return "".join([prefix, files[targetX], str(8 - targetY)])

    # Moves piece on the board, returns true if move was completed (and consequently legal) and false if move was not able
    # to be completed (move was illegal or the coordinates provided did not contain a piece). potentialMove is from the POV of
    # the current player
    def Move(self, potentialMove):
        # If it is the first move, need to generate the set of all legal moves for white (if not first move, this is done at the)
        # end of this function
        if (self.totalMoves == 1):
            self.gameState = self.GameOver()

        # if (self.whiteToMove == True):
        #     print ("White's Moves:" , self.allLegalMoves)
        # else:
        #     print ("Black's Moves: ", self.allLegalMoves)

        # If the game isn't over at this moment, then proceed in determining whether the move is legal or not
        if (self.gameState == 0):
            if (potentialMove in self.allLegalMoves):
                takes = False
                move = np.unravel_index(potentialMove, (8,8,73))
                moveInfo = self.moveDecoder[move[2]] # Tuple of (dy, dx) for move

                # Piece to be moved
                piece = self.board[move[0]][move[1]]
                # If it's blacks move, the coordinates and moveInfo are mirror image so fix them
                if (self.whiteToMove == False):
                    piece = self.board[7-move[0]][7-move[1]]
                    moveInfo = (moveInfo[0] * -1, moveInfo[1] * -1)

                # Check for captures
                if (isinstance(self.board[piece.rankNum + moveInfo[0]][piece.fileNum + moveInfo[1]], Piece)):
                    takes = True

                # If piece isn't of the right color (i.e. it is a black piece when it's white to move or vice versa) return False
                if (piece.color != self.whiteToMove):
                    return False

                # Move is legal, add it to the algebraic representation of the game
                self.lastMove = self.CoordToAlgebraic(move, takes)
                

                # If piece is a king and it is castling (abs(moveInfo[1])) == 2 
                if (isinstance(piece, King) and abs(moveInfo[1]) == 2):
                    # Castling
                    # Dx of 2 from white POV mean's go east, to go east from black's POV dx is -2
                    if (moveInfo[1] == 2):
                        # If piece is white and dx is 2, move is kingside castle, so move rook as well
                        if (piece.color == True):
                            # Move white king
                            self.board[piece.rankNum][piece.fileNum] = None
                            self.board[7][6] = piece
                            piece.rankNum = 7
                            piece.fileNum = 6
                            self.whiteKing = (7, 6)
                            # Change hasMoved field since king has moved
                            piece.hasMoved = True

                            # Move kingside white rook
                            self.board[7][5] = self.board[7][7]
                            self.board[7][5].rankNum = 7
                            self.board[7][5].fileNum = 5
                            self.board[7][5].hasMoved = True
                            self.board[7][7] = None
                        else:
                            # Move the King
                            self.board[piece.rankNum][piece.fileNum] = None
                            self.board[0][6] = piece
                            piece.rankNum = 0
                            piece.fileNum = 6
                            self.blackKing = (0, 6)
                            # Change hasMoved field since king has moved
                            piece.hasMoved = True

                            # Move kingside black rook
                            self.board[0][5] = self.board[0][7]
                            self.board[0][5].rankNum = 0
                            self.board[0][5].fileNum = 5
                            self.board[0][5].hasMoved = True
                            self.board[0][7] = None
                            
                    # Dx of -2 from white POV mean's go west, to go west from black POV dx is +2
                    elif (moveInfo[1] == -2):
                        if (piece.color == True):
                            # Move white king
                            self.board[piece.rankNum][piece.fileNum] = None
                            self.board[7][2] = piece
                            piece.rankNum = 7
                            piece.fileNum = 2
                            self.whiteKing = (7, 2)
                            # Change hasMoved field since the king has moved
                            piece.hasMoved = True

                            # Move queenside white rook
                            self.board[7][3] = self.board[7][0]
                            self.board[7][3].rankNum = 7
                            self.board[7][3].fileNum = 3
                            self.board[7][3].hasMoved = True
                            self.board[7][0] = None
                        
                        # If piece is black and dx is 2, move is kingside castle, so move kingside black rook
                        else:
                            # Move black king
                            self.board[piece.rankNum][piece.fileNum] = None
                            self.board[0][2] = piece
                            piece.rankNum = 0
                            piece.fileNum = 2
                            self.blackKing = (0, 2)
                            # Change hasMoved field since king has moved
                            piece.hasMoved = True

                            # Move queenside black rook
                            self.board[0][3] = self.board[0][0]
                            self.board[0][3].rankNum = 0
                            self.board[0][3].fileNum = 3
                            self.board[0][3].hasMoved = True
                            self.board[0][0] = None

                # Piece is white pawn about to be promoted (rankNum == 1)
                elif (isinstance(piece, Pawn) and piece.color == True and piece.rankNum == 1):
                    # Queen promotions
                    if (move[2] == 0 or move[2] == 7 or move[2] == 49):
                        self.board[piece.rankNum + moveInfo[0]][piece.fileNum+moveInfo[1]] = Queen (True, piece.rankNum + moveInfo[0], piece.fileNum+moveInfo[1], self)
                    # Knight promotions
                    elif (move[2] == 64 or move[2] == 67 or move[2] == 70):
                        self.board[piece.rankNum + moveInfo[0]][piece.fileNum+moveInfo[1]] = Knight (True, piece.rankNum + moveInfo[0], piece.fileNum+moveInfo[1], self)
                    # Bishop promotions
                    elif (move[2] == 65 or move[2] == 68 or move[2] == 71):
                        self.board[piece.rankNum + moveInfo[0]][piece.fileNum+moveInfo[1]] = Bishop (True, piece.rankNum + moveInfo[0], piece.fileNum+moveInfo[1], self)
                    # Rook promotions
                    elif (move[2] == 66 or move[2] == 69 or move[2] == 72):
                        self.board[piece.rankNum + moveInfo[0]][piece.fileNum+moveInfo[1]] = Rook (True, piece.rankNum + moveInfo[0], piece.fileNum+moveInfo[1], self)  
                    self.board[piece.rankNum][piece.fileNum] = None
                # Piece is black pawn about to be promoted (rankNum == 6)
                elif (isinstance(piece, Pawn) and piece.color == False and piece.rankNum == 6):
                    # Queen promotions
                    if (move[2] == 0 or move[2] == 7 or move[2] == 49):
                        self.board[piece.rankNum + moveInfo[0]][piece.fileNum+moveInfo[1]] = Queen (False, piece.rankNum + moveInfo[0], piece.fileNum+moveInfo[1], self)
                    # Knight promotions
                    elif (move[2] == 64 or move[2] == 67 or move[2] == 70):
                        self.board[piece.rankNum + moveInfo[0]][piece.fileNum+moveInfo[1]] = Knight (False, piece.rankNum + moveInfo[0], piece.fileNum+moveInfo[1], self)
                    # Bishop promotions
                    elif (move[2] == 65 or move[2] == 68 or move[2] == 71):
                        self.board[piece.rankNum + moveInfo[0]][piece.fileNum+moveInfo[1]] = Bishop (False, piece.rankNum + moveInfo[0], piece.fileNum+moveInfo[1], self)
                    # Rook promotions
                    elif (move[2] == 66 or move[2] == 69 or move[2] == 72):
                        self.board[piece.rankNum + moveInfo[0]][piece.fileNum+moveInfo[1]] = Rook (False, piece.rankNum + moveInfo[0], piece.fileNum+moveInfo[1], self)  
                    self.board[piece.rankNum][piece.fileNum] = None            
                else:
                    if (isinstance(piece, (King, Rook, Pawn)) and piece.hasMoved == False):
                        piece.hasMoved = True
                    
                    # En passant moves, diagonal move will result in pawn "capturing" on empty square, need to remove the
                    # pawn that was caputred en passant
                    if (isinstance(piece, Pawn) and (move[2] == 49 or move[2] == 7) and self.board[piece.rankNum + moveInfo[0]][piece.fileNum+moveInfo[1]] is None):
                        if (piece.color == True):
                            takes = True
                            self.lastMove = self.CoordToAlgebraic(move, takes)
                            self.board[piece.rankNum + moveInfo[0]][piece.fileNum + moveInfo[1]] = piece
                            self.board[piece.rankNum + moveInfo[0] + 1][piece.fileNum + moveInfo[1]] = None
                            self.board[piece.rankNum][piece.fileNum] = None
                            piece.rankNum = piece.rankNum + moveInfo[0]
                            piece.fileNum = piece.fileNum + moveInfo[1]
                        else:
                            takes = True
                            self.lastMove = self.CoordToAlgebraic(move, takes)
                            self.board[piece.rankNum + moveInfo[0]][piece.fileNum + moveInfo[1]] = piece
                            self.board[piece.rankNum + moveInfo[0] - 1][piece.fileNum + moveInfo[1]] = None
                            self.board[piece.rankNum][piece.fileNum] = None
                            piece.rankNum = piece.rankNum + moveInfo[0]
                            piece.fileNum = piece.fileNum + moveInfo[1]
                    else:
                        # Move piece and update rankNum and fileNum fields
                        self.board[piece.rankNum + moveInfo[0]][piece.fileNum + moveInfo[1]] = piece
                        self.board[piece.rankNum][piece.fileNum] = None
                        # Update coordinate fields
                        piece.rankNum = piece.rankNum + moveInfo[0]
                        piece.fileNum = piece.fileNum + moveInfo[1]

                    # If piece to be moved is a king, update the appropriate whiteKing or blackKing board fields
                    if (isinstance(piece, King) == True):
                        if (piece.color == True):
                            self.whiteKing = (piece.rankNum, piece.fileNum)
                        else:
                            self.blackKing = (piece.rankNum, piece.fileNum)

                    

                if (self.whiteToMove == True):
                    self.totalMoves +=1
                
                if (takes == True or isinstance(piece, Pawn)):
                    self.lastCapture = self.totalMoves

                self.whiteToMove = not self.whiteToMove
                self.listOfMoves.append(self.lastMove)
                # Test if game is over for the opposing side (and thereby generating legal moves for that side)
                self.gameState = self.GameOver()
                return True
            else:
                return False
        else:
            return False

    # Returns -1 if game ends in stalemate for side (white or black) that the function is called with
    #  or drawn by insufficient material on the board, returns 1 if game ends in checkmate
    #  for the side that the function is called with, and returns 0 if the game continues.
    def GameOver(self):
        # 50 move rule for draws, if 50 moves have passed since last capture or pawn move game is automatically drawn
        if (self.totalMoves - self.lastCapture >= 50):  
            return -1

        # Set legalMoves to empty set, then calculate all legal moves
        self.allLegalMoves = set()

        whiteRookCount = 0
        whiteKnightCount = 0
        whiteBishopCount = 0
        whitePawnCount = 0
        whiteQueenCount = 0

        blackRookCount = 0
        blackKnightCount = 0
        blackBishopCount = 0
        blackPawnCount = 0
        blackQueenCount = 0

        kingCoords = ()
        if (self.whiteToMove == True):
            kingCoords = self.whiteKing
        else:
            kingCoords = self.blackKing

        inCheck = InCheck(self.board, kingCoords)
        # Count pieces and check to see whether piece can move
        canMove = False
        for row in self.board:
            for piece in row:
                if (isinstance(piece, Piece)):
                    if (piece.color == self.whiteToMove and len(piece.MoveList()) != 0):
                        canMove = True
                        self.allLegalMoves = self.allLegalMoves | piece.legalMoves
                if (isinstance(piece, Queen)):
                    if (piece.color == True):
                        whiteQueenCount += 1
                    else:
                        blackQueenCount += 1
                elif (isinstance(piece, Bishop)):
                    if (piece.color == True):
                        whiteBishopCount += 1
                    else:
                        blackBishopCount += 1
                elif (isinstance(piece, Rook)):
                    if (piece.color == True):
                        whiteRookCount += 1
                    else:
                        blackRookCount += 1
                elif (isinstance(piece, Knight)):
                    if (piece.color == True):
                        whiteKnightCount += 1
                    else:
                        blackKnightCount += 1
                elif (isinstance(piece, Pawn)):
                    if (piece.color == True):
                        whitePawnCount += 1
                    else:
                        blackPawnCount += 1

        if (canMove == False):
            # If no piece of the side passed can move and the king is in check, it is checkmate
            if (inCheck == True):
                self.listOfMoves[-1] = "".join([self.listOfMoves[-1], "#"])
                return 1
            # If no piece of the side passed can move and the king is not in check, it is stalemate
            else:
                self.listOfMoves[-1] = "".join([self.listOfMoves[-1], "1/2-1/2"])
                return -1

        # Check if potential insufficient material remains, if there are pawns, rooks, or queens on the board then the game
        # cannot be drawn by insufficient material
        insufficientMatRemaining = False
        if ((whiteQueenCount == 0 and blackQueenCount == 0) and (whiteRookCount == 0 and blackRookCount == 0) and (whitePawnCount == 0 and blackPawnCount == 0)):
            insufficientMatRemaining = True
        else:
            if (inCheck == True):
                self.listOfMoves[-1] = "".join([self.listOfMoves[-1], "+"])
            return 0

        if (insufficientMatRemaining == True):
            # Game is drawn if there are no bishops and there are ONLY 1 or less knights on either side in the gameq
            if ((whiteBishopCount == 0 and blackBishopCount == 0) and (whiteKnightCount < 2 and blackKnightCount < 2)):
                self.listOfMoves[-1] = "".join([self.listOfMoves[-1], "1/2-1/2"])
                return -1
            # Game is drawn if there are no knights and there are ONLY 1 or less knights on either side in the game
            if ((whiteKnightCount == 0 and blackKnightCount == 0) and (whiteBishopCount < 2 and blackBishopCount < 2)):
                self.listOfMoves[-1] = "".join([self.listOfMoves[-1], "1/2-1/2"])
                return -1
            # Game is drawn if there is only 2 knights vs a king
            if ((whiteKnightCount == 2 and blackBishopCount == 0) or (whiteBishopCount == 0 and blackKnightCount == 2)):
                self.listOfMoves[-1] = "".join([self.listOfMoves[-1], "1/2-1/2"])
                return -1
            # Game is drawn if it's just 1 white knight vs only one black bishop
            if ((whiteKnightCount == 1 and whiteBishopCount == 0) and (blackBishopCount == 1 and blackKnightCount == 0)):
                self.listOfMoves[-1] = "".join([self.listOfMoves[-1], "1/2-1/2"])
                return -1
            # Game is drawn if it's just one white bishop vs only one black knight
            if ((whiteBishopCount == 1 and whiteKnightCount == 0) and (blackBishopCount == 0 and blackKnightCount == 1)):
                self.listOfMoves[-1] = "".join([self.listOfMoves[-1], "1/2-1/2"])
                return -1

        # If none of the previous conditions are met, the game continues
        return 0

    # Prints a character representation of the board in which black pieces are lowercase and white pieces are uppercase
    def ShowBoard(self):
        print(" -----------------")
        for row in range(0, 8):
            for col in range(0, 8):
                if (col == 0):
                    print("| ", end="")
                piece = self.board[row][col]
                
                if (isinstance(piece, Pawn) == True):
                    if (piece.color == True):
                        print("P ", end="")
                    else:
                        print("p ", end="")
                elif (isinstance(piece, Queen) == True):
                    if (piece.color == True):
                        print("Q ", end="")
                    else:
                        print("q ", end="")
                elif (isinstance(piece, Rook) == True):
                    if (piece.color == True):
                        print("R ", end="")
                    else:
                        print ("r ", end = "")
                elif (isinstance(piece, Bishop) == True):
                    if(piece.color == True):
                        print("B ", end="")
                    else:
                        print("b ", end="")
                elif (isinstance(piece, Knight) == True):
                    if (piece.color == True):
                        print("N ", end="")
                    else:
                        print ("n ", end = "")
                elif (isinstance(piece, King) == True):
                    if (piece.color == True):
                        print("K ", end="")
                    else:
                        print ("k ", end = "")
                else:
                    print("  ", end="")
                if (col == 7):
                    print("|")
        print(" -----------------")


