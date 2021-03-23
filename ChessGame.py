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
    if (board.gameState == 1):
        return 0

    # Index to piece 0 : Pawn, 1 : Knight, 2 : Bishop, 3 : Rook, 4 : Queen, 5: King
    PIECE_SQUARES = []
    if (board.totalMoves < 20):
        if (color == True):
            PIECE_SQUARES = [WHITE_PAWN_MG, WHITE_KNIGHT_MG, WHITE_BISHOP_MG, WHITE_ROOK_MG, WHITE_QUEEN_MG, WHITE_KING_MG]
        else:
            PIECE_SQUARES = [BLACK_PAWN_MG, BLACK_KNIGHT_MG, BLACK_BISHOP_MG, BLACK_ROOK_MG, BLACK_QUEEN_MG, BLACK_KING_MG]
    else:
        if (color == True):
            PIECE_SQUARES = [WHITE_PAWN_EG, WHITE_KNIGHT_EG, WHITE_BISHOP_EG, WHITE_ROOK_EG, WHITE_QUEEN_EG, WHITE_KING_EG]
        else:
            PIECE_SQUARES = [BLACK_PAWN_EG, BLACK_KNIGHT_EG, BLACK_BISHOP_EG, BLACK_ROOK_EG, BLACK_QUEEN_EG, BLACK_KING_EG]

    STACKED_PENALTY = 0
    if (board.totalMoves < 15):
        STACKED_PENALTY = 2
    else:
        STACKED_PENALTY = 1.9

    evaluation = 0

    for col in range(0, 8):
        pawnCount = 0
        for row in range(0, 8):
            if (isinstance(board.board[row][col], Queen) and board.board[row][col].color == color):
                evaluation += QUEEN_VAL + PIECE_SQUARES[4][row][col]
            elif (isinstance(board.board[row][col], Bishop) and board.board[row][col].color == color):
                evaluation += BISHOP_VAL + PIECE_SQUARES[2][row][col]
            elif (isinstance(board.board[row][col], Rook) and board.board[row][col].color == color):
                evaluation += ROOK_VAL + PIECE_SQUARES[3][row][col]
            elif (isinstance(board.board[row][col], Knight) and board.board[row][col].color == color):
                evaluation += KNIGHT_VAL + PIECE_SQUARES[1][row][col]
            elif (isinstance(board.board[row][col], Pawn) and board.board[row][col].color == color):
                evaluation += PAWN_VAL + + PIECE_SQUARES[0][row][col]
                pawnCount += 1
            elif (isinstance(board.board[row][col], King) and board.board[row][col].color == color):
                evaluation += PIECE_SQUARES[5][row][col]
        if (pawnCount > 1):
            print(pawnCount)
            if (board.totalMoves < 15):
                evaluation -= (STACKED_PENALTY ** pawnCount-1) * 20
            else:
                evaluation -= (STACKED_PENALTY ** pawnCount) * 15

    return evaluation


# Determines if a square is in Check or not given the board it is placed on and its coordinates
def InCheck(board, coordinates):
    rankNum, fileNum = coordinates

    # Check for pawn checks on White King
    if (isinstance(board[rankNum][fileNum], Piece) == True and board[rankNum][fileNum].color == True):
        for dx in (-1, 1):
            # Impossible for White King to be in check from a pawn if the white king is on the 7th or 8th ranks
            if (rankNum > 1):
                pawnFile = fileNum + dx
                if (pawnFile >= 0 and pawnFile <= 7):
                    if (isinstance(board[rankNum-1][pawnFile], Pawn) == True and board[rankNum-1][pawnFile].color != board[rankNum][fileNum].color):
                        return True
            else:
                break
    # Check for pawn checks on Black King
    elif (isinstance(board[rankNum][fileNum], Piece) == True and board[rankNum][fileNum].color == False):
        for dx in (-1, 1):
            # Impossible for Black King to be in check from a pawn if the black king is on the 1st or 2nd ranks
            if (rankNum < 6):
                pawnFile = fileNum + dx
                if (pawnFile >= 0 and pawnFile <= 7):
                    if (isinstance(board[rankNum+1][pawnFile], Pawn) == True and board[rankNum+1][pawnFile].color != board[rankNum][fileNum].color):
                        return True
            else:
                break

    # Checks for checks from enemy king (not technically possible, but necessary to generate legal king moves)
    for dy, dx in ((1, 1), (1, -1), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 0), (-1, 1)):
        targetY = rankNum + dy
        targetX = fileNum + dx
        if ((targetX >= 0 and targetX <= 7) and (targetY >= 0 and targetY <= 7)):
            if (isinstance(board[targetY][targetX], King) == True):
                return True

    # Checks for checks from Knights
    for dy, dx in ((-1, 2), (-1, -2), (1, 2), (1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)):
        targetY = rankNum + dy
        targetX = fileNum + dx
        if ((targetY >= 0 and targetY <= 7) and (targetX >= 0 and targetX <= 7)):
            if (isinstance(board[targetY][targetX], Knight) == True and board[targetY][targetX].color != board[rankNum][fileNum].color):
                return True

    # Check for vertical checks from rooks/queens
    for y in range(rankNum - 1, -1, -1):
        if (isinstance(board[y][fileNum], Piece) == True and board[y][fileNum].color != board[rankNum][fileNum].color):
            if (isinstance(board[y][fileNum], Rook) == True):
                return True
            else:
                break
        elif (isinstance(board[y][fileNum], Piece) == True and board[y][fileNum].color == board[rankNum][fileNum].color):
            break

    for y in range(rankNum + 1, 8):
        if (isinstance(board[y][fileNum], Piece) == True and board[y][fileNum].color != board[rankNum][fileNum].color):
            if (isinstance(board[y][fileNum], Rook) == True):
                return True
            else:
                break
        elif (isinstance(board[y][fileNum], Piece) == True and board[y][fileNum].color == board[rankNum][fileNum].color):
            break

    # Check for horizontal checks from rooks/queens
    for x in range(fileNum-1, -1, -1):
        if (isinstance(board[rankNum][x], Piece) == True and board[rankNum][x].color != board[rankNum][fileNum].color):
            if (isinstance(board[rankNum][x], Rook) == True):
                return True
            else:
                break
        elif (isinstance(board[rankNum][x], Piece) == True and board[rankNum][x].color == board[rankNum][fileNum].color):
            break

    for x in range(fileNum + 1, 8):
        if (isinstance(board[rankNum][x], Piece) == True and board[rankNum][x].color != board[rankNum][fileNum].color):
            if (isinstance(board[rankNum][x], Rook) == True):
                return True
            else:
                break
        elif (isinstance(board[rankNum][x], Piece) == True and board[rankNum][x].color == board[rankNum][fileNum].color):
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
            if (isinstance(board[y-1][x-1], Bishop) == True):
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
            if (isinstance(board[y-1][x+1], Bishop) == True):
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
            if (isinstance(board[y+1][x-1], Bishop) == True):
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
            if (isinstance(board[y+1][x+1], Bishop) == True):
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

    def CoordToAlgebraic(self, coordinate, promotion="", takes=False, toLeft=False):
        rankNum, fileNum = coordinate

        # If king has castled
        if (isinstance(self, King) == True):
            # Queenside castle
            if (fileNum-self.fileNum == -2):
                return "0-0-0"
            elif (fileNum-self.fileNum == 2):
                return "0-0"

        files = {
            0: "a",
            1: "b",
            2: "c",
            3: "d",
            4: "e",
            5: "f",
            6: "g",
            7: "h"
        }

        prefix = ""
        if (isinstance(self, Queen) == True):
            prefix = "Q"
        elif (isinstance(self, Bishop) == True):
            prefix = "B"
        elif (isinstance(self, Knight) == True):
            prefix = "N"
        elif (isinstance(self, King) == True):
            prefix = "K"
        elif (isinstance(self, Rook) == True):
            prefix = "R"

        if (takes == True):
            if (isinstance(self, Pawn) == False):
                return prefix + "x" + str(files[fileNum]) + str(abs(rankNum-7) + 1)
            if (promotion != "" and isinstance(self, Pawn) == True):
                if (toLeft == True):
                    return str(files[fileNum+1]) + "x" + str(files[fileNum]) + str(abs(rankNum-7) + 1) + promotion
                else:
                    return str(files[fileNum-1]) + "x" + str(files[fileNum]) + str(abs(rankNum-7) + 1) + promotion
            else:
                if (toLeft == True):
                    return str(files[fileNum+1]) + "x" + str(files[fileNum]) + str(abs(rankNum-7) + 1)
                else:
                    return str(files[fileNum-1]) + "x" + str(files[fileNum]) + str(abs(rankNum-7) + 1)
        else:
            if (isinstance(self, Pawn) == True and promotion != ""):
                return str(files[fileNum]) + str(abs(rankNum-7) + 1) + promotion
            return prefix + str(files[fileNum]) + str(abs(rankNum-7) + 1)


class Pawn(Piece):
    hasMoved = False

    def MoveList(self):
        moves = set()
        kingCoord = ()
        if (self.color == True):
            kingCoord = self.board.whiteKing
        else:
            kingCoord = self.board.blackKing

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
                    moves.add((self.rankNum-1, self.fileNum))

                # Change board back to original state
                self.board.board[self.rankNum-1][self.fileNum] = None
                self.board.board[self.rankNum][self.fileNum] = self

            # If diagonal pawn capture squares contain black pieces, pawn can move there
            if (self.fileNum - 1 >= 0 and isinstance(self.board.board[self.rankNum-1][self.fileNum-1], Piece) == True and self.board.board[self.rankNum-1][self.fileNum-1].color != self.color):
                piece = self.board.board[self.rankNum-1][self.fileNum-1]

                # Change board as if the potential move was played
                self.board.board[self.rankNum-1][self.fileNum-1] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective side's king is in check after the move
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    moves.add((self.rankNum-1, self.fileNum-1))

                # Change board back to original state
                self.board.board[self.rankNum-1][self.fileNum-1] = piece
                self.board.board[self.rankNum][self.fileNum] = self

            if (self.fileNum + 1 <= 7 and isinstance(self.board.board[self.rankNum-1][self.fileNum+1], Piece) == True and self.board.board[self.rankNum-1][self.fileNum+1].color != self.color):
                piece = self.board.board[self.rankNum-1][self.fileNum+1]

                # Change board as if the potential move was played
                self.board.board[self.rankNum-1][self.fileNum+1] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective side's king is in check after the move
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    moves.add((self.rankNum-1, self.fileNum+1))

                # Change board back to original state
                self.board.board[self.rankNum-1][self.fileNum+1] = piece
                self.board.board[self.rankNum][self.fileNum] = self

            # Check for en passant moves
            if (self.rankNum == 3 and len(self.board.lastMove) == 2 and int(self.board.lastMove[-1]) == 5):
                if (ord(self.board.lastMove[0]) - 97 == self.fileNum-1):
                    moves.add((2, self.fileNum-1))
                if (ord(self.board.lastMove[0]) - 97 == self.fileNum+1):
                    moves.add((2, self.fileNum+1))
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
                    moves.add((self.rankNum+1, self.fileNum))

                # Change board back to original state
                self.board.board[self.rankNum+1][self.fileNum] = None
                self.board.board[self.rankNum][self.fileNum] = self
            # If diagonal pawn capture squares contain black pieces, pawn can move there
            if (self.fileNum - 1 >= 0 and isinstance(self.board.board[self.rankNum+1][self.fileNum-1], Piece) == True and self.board.board[self.rankNum+1][self.fileNum-1].color != self.color):
                piece = self.board.board[self.rankNum+1][self.fileNum-1]

                # Change board as if the potential move was played
                self.board.board[self.rankNum+1][self.fileNum-1] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective side's king is in check after the move
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    moves.add((self.rankNum+1, self.fileNum-1))

                # Change board back to original state
                self.board.board[self.rankNum+1][self.fileNum-1] = piece
                self.board.board[self.rankNum][self.fileNum] = self

            if (self.fileNum + 1 <= 7 and isinstance(self.board.board[self.rankNum+1][self.fileNum+1], Piece) == True and self.board.board[self.rankNum+1][self.fileNum+1].color != self.color):
                piece = self.board.board[self.rankNum+1][self.fileNum+1]

                # Change board as if the potential move was played
                self.board.board[self.rankNum+1][self.fileNum+1] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective side's king is in check after the move
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    moves.add((self.rankNum+1, self.fileNum+1))

                # Change board back to original state
                self.board.board[self.rankNum+1][self.fileNum+1] = piece
                self.board.board[self.rankNum][self.fileNum] = self

                # Check for en passant moves
            if (self.rankNum == 4 and len(self.board.lastMove) == 2 and int(self.board.lastMove[-1]) == 4):
                if (ord(self.board.lastMove[0]) - 97 == self.fileNum-1):
                    moves.add((5, self.fileNum-1))
                if (ord(self.board.lastMove[0]) - 97 == self.fileNum+1):
                    moves.add((5, self.fileNum+1))
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
                        moves.add((4, self.fileNum))

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
                        moves.add((3, self.fileNum))

                    # Change board back to original state
                    self.board.board[3][self.fileNum] = None
                    self.board.board[self.rankNum][self.fileNum] = self
        self.legalMoves = moves
        return moves


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
        moves = set()
        for dy, dx in ((1, 1), (1, -1), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 0), (-1, 1)):
            targetY = self.rankNum + dy
            targetX = self.fileNum + dx
            if ((targetX >= 0 and targetX <= 7) and (targetY >= 0 and targetY <= 7)):
                if (isinstance(self.board.board[targetY][targetX], Piece) == True and self.board.board[targetY][targetX].color == self.color):
                    continue
                else:
                    # Change board as if potential move had been played
                    piece = self.board.board[targetY][targetX]
                    self.board.board[targetY][targetX] = self
                    self.board.board[self.rankNum][self.fileNum] = None

                    # Check to see if the king is in check after the move has been played
                    inCheck = InCheck(self.board.board, (targetY, targetX))
                    if (inCheck != True):
                        moves.add((targetY, targetX))

                    # Change board to original state
                    self.board.board[targetY][targetX] = piece
                    self.board.board[self.rankNum][self.fileNum] = self

        # Check to see if white king can castle
        if (self.color == True and self.hasMoved == False):
            # Check kingside castle
            if (isinstance(self.board.board[7][7], Rook) == True and self.board.board[7][7].hasMoved == False):
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
                        moves.add((7, 6))
            # Check queenside castle
            if (isinstance(self.board.board[7][0], Rook) == True and self.board.board[7][0].hasMoved == False):
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
                        moves.add((7, 2))

        # Check to see if black king can castle
        elif (self.color == False and self.hasMoved == False):
            # Check kingside castle
            if (isinstance(self.board.board[0][7], Rook) == True and self.board.board[0][7].hasMoved == False):
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
                        moves.add((0, 6))
            # Check queenside castle
            if (isinstance(self.board.board[0][0], Rook) == True and self.board.board[0][0].hasMoved == False):
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
                        moves.add((0, 2))

        self.legalMoves = moves
        return moves


class Bishop(Piece):
    def MoveList(self):
        moves = set()
        kingCoord = ()
        if (self.color == True):
            kingCoord = self.board.whiteKing
        else:
            kingCoord = self.board.blackKing

        x = self.fileNum
        y = self.rankNum
        # Start by checking the legal diagonal moves in the north west direction
        while (x > 0 and y > 0):
            # If the potential square is empty, piece can move there
            if (self.board.board[y-1][x-1] is None):
                # Change board as if potential move had been played
                self.board.board[y-1][x-1] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective king is in check
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    moves.add((y-1, x-1))

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
                    moves.add((y-1, x-1))

                # Change board back to original state
                self.board.board[y-1][x-1] = piece
                self.board.board[self.rankNum][self.fileNum] = self
                break
            # If the potential square has a piece of the same color, bishop can not occupy that square or
            # go further on the diagonal, so break
            elif (self.board.board[y-1][x-1].color == self.color):
                break

        x = self.fileNum
        y = self.rankNum

        # Check the legal diagonal moves in the nort east direction
        while (x < 7 and y > 0):
            if (self.board.board[y-1][x+1] is None):
                # Change board as if potential move had been played
                self.board.board[y-1][x+1] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective king is in check
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    moves.add((y-1, x+1))

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
                    moves.add((y-1, x+1))

                # Change board back to original state
                self.board.board[y-1][x+1] = piece
                self.board.board[self.rankNum][self.fileNum] = self
                break
            elif (self.board.board[y-1][x+1].color == self.color):
                break

        x = self.fileNum
        y = self.rankNum

        # Check the legal diagonal moves in the south west direction
        while (x > 0 and y < 7):
            if (self.board.board[y+1][x-1] is None):
                # Change board as if potential move had been played
                self.board.board[y+1][x-1] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective king is in check
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    moves.add((y+1, x-1))

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
                    moves.add((y+1, x-1))

                # Change board back to original state
                self.board.board[y+1][x-1] = piece
                self.board.board[self.rankNum][self.fileNum] = self
                break
            elif (self.board.board[y+1][x-1].color == self.color):
                break

        x = self.fileNum
        y = self.rankNum

        # Check the legal diagonal moves in the south east direction
        while (x < 7 and y < 7):
            if (self.board.board[y+1][x+1] is None):
                # Change board as if potential move had been played
                self.board.board[y+1][x+1] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective king is in check
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    moves.add((y+1, x+1))

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
                    moves.add((y+1, x+1))

                # Change board back to original state
                self.board.board[y+1][x+1] = piece
                self.board.board[self.rankNum][self.fileNum] = self
                break
            elif (self.board.board[y+1][x+1].color == self.color):
                break

        self.legalMoves = moves
        return moves


class Knight(Piece):
    def MoveList(self):
        moves = set()
        kingCoord = ()
        if (self.color == True):
            kingCoord = self.board.whiteKing
        else:
            kingCoord = self.board.blackKing

        for dy, dx in ((-1, 2), (-1, -2), (1, 2), (1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)):
            newRank = self.rankNum + dy
            newFile = self.fileNum + dx
            # Test that possible knight move is still in the self.board.board.board
            if ((newRank >= 0 and newRank <= 7) and (newFile >= 0 and newFile <= 7)):
                # If the potential knight move is on a empty square, it's legal
                if (self.board.board[newRank][newFile] is None):

                    self.board.board[newRank][newFile] = self
                    self.board.board[self.rankNum][self.fileNum] = None

                    inCheck = InCheck(self.board.board, (kingCoord))
                    if (inCheck != True):
                        moves.add((newRank, newFile))

                    self.board.board[newRank][newFile] = None
                    self.board.board[self.rankNum][self.fileNum] = self
                # If an enemy piece is on the potential move's square, the move is legal and the knight
                # can take the piece
                elif (self.board.board[newRank][newFile].color != self.color):
                    piece = self.board.board[newRank][newFile]

                    self.board.board[newRank][newFile] = self
                    self.board.board[self.rankNum][self.fileNum] = None
                    inCheck = InCheck(self.board.board, (kingCoord))
                    if (inCheck != True):
                        moves.add((newRank, newFile))

                    self.board.board[newRank][newFile] = piece
                    self.board.board[self.rankNum][self.fileNum] = self
                # If a piece of the same color is on the potential move's square, do nothing (don't add it to the set
                # of possible moves since it's not legal)

        self.legalMoves = moves
        return moves


class Rook (Piece):
    hasMoved = False

    def MoveList(self):
        moves = set()
        kingCoord = ()
        if (self.color == True):
            kingCoord = self.board.whiteKing
        else:
            kingCoord = self.board.blackKing

        # Check all legal vertical moves from rook's current position to the left
        for y in range(self.rankNum - 1, -1, -1):
            # If potential position is empty, rook can move there
            if (self.board.board[y][self.fileNum] is None):
                # Change board as if potential move had been played
                self.board.board[y][self.fileNum] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective side king is in check after the move has been played
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    moves.add((y, self.fileNum))

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
                    moves.add((y, self.fileNum))

                # Change board back to original state
                self.board.board[y][self.fileNum] = piece
                self.board.board[self.rankNum][self.fileNum] = self
                break
            # If potential position has a piece of the same color, rook can't go there, so break
            elif (self.board.board[y][self.fileNum].color == self.color):
                break

        # Check all legal vertical moves from rook's current position to the right
        for y in range(self.rankNum + 1, 8):
            if (self.board.board[y][self.fileNum] is None):
                # Change board as if potential move had been played
                self.board.board[y][self.fileNum] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective side king is in check after the move has been played
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    moves.add((y, self.fileNum))

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
                    moves.add((y, self.fileNum))

                # Change board back to original state
                self.board.board[y][self.fileNum] = piece
                self.board.board[self.rankNum][self.fileNum] = self
                break
            elif (self.board.board[y][self.fileNum].color == self.color):
                break

        # Check all legal horizontal moves from rook's current position to the bottom of the self.board.board
        for x in range(self.fileNum - 1, -1, -1):
            if (self.board.board[self.rankNum][x] is None):
                # Change board as if potential move had been played
                self.board.board[self.rankNum][x] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective side king is in check after the move has been played
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    moves.add((self.rankNum, x))

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
                    moves.add((self.rankNum, x))

                # Change board back to original state
                self.board.board[self.rankNum][x] = piece
                self.board.board[self.rankNum][self.fileNum] = self
                break
            elif (self.board.board[self.rankNum][x].color == self.color):
                break

        # Check all legal horizontal moves from the rook's current position to the top of the self.board.board
        for x in range(self.fileNum + 1, 8):
            if (self.board.board[self.rankNum][x] is None):
                # Change board as if potential move had been played
                self.board.board[self.rankNum][x] = self
                self.board.board[self.rankNum][self.fileNum] = None

                # Check to see if respective side king is in check after the move has been played
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    moves.add((self.rankNum, x))

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
                    moves.add((self.rankNum, x))

                # Change board back to original state
                self.board.board[self.rankNum][x] = piece
                self.board.board[self.rankNum][self.fileNum] = self
                break
            elif (self.board.board[self.rankNum][x].color == self.color):
                break
        self.legalMoves = moves
        return moves


class Queen(Bishop, Rook):
    def MoveList(self):
        # Queen moves are just a combination of the set of possible rook moves and possible bishop moves.
        self.legalMoves = Bishop.MoveList(self) | Rook.MoveList(self)
        return self.legalMoves


class Board ():
    listOfMoves = []
    lastMove = ""
    totalMoves = 0
    whiteToMove = True
    gameState = 0
    lastCapture = 0

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
    
    # Resets the board entirely, including all its data fields
    def InitializeBoard (self):
        self.listOfMoves = []
        self.lastMove = ""
        self.totalMoves = 0
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

    # Moves piece on the board, returns true if move was completed (and consequently legal) and false if move was not able
    # to be completed (move was illegal or the coordinates provided did not contain a piece)
    def Move(self, lastMove, promotion = ""):
        print (lastMove)
        curCoords = lastMove[0]
        targetCoords = lastMove[1]
        curY, curX = curCoords
        targetY, targetX = targetCoords

        # If piece selected is an empty square, return False; you can't move an empty square
        if (self.board[curY][curX] is None):
            return False

        # If a piece of the wrong color (not currently that color's move) tries to move, return False since it is not
        # that side's turn
        if (self.board[curY][curX].color != self.whiteToMove):
            return False

        moves = self.board[curY][curX].legalMoves
        # print (moves)
        if (targetCoords in moves):
            didTake = False
            toLeft = False
            # If a piece was taken, set the didTake variable to True (used for algebraic notation of the game)
            if (isinstance(self.board[targetY][targetX], Piece) == True):
                if (self.board[targetY][targetX].color != self.board[curY][curX].color):
                    didTake = True

            # Check for en passant captures
            if (isinstance(self.board[curY][curX], Pawn) == True):
                # En passant for white pawns
                if ((curY-targetY == 1) and (abs(curX-targetX) == 1) and self.board[targetY][targetX] == None):
                    didTake = True
                    self.board[targetY+1][targetX] = None
                # En passant for black pawns
                if ((curY-targetY == -1) and (abs(curX-targetX) == 1) and self.board[targetY][targetX] == None):
                    didTake = True
                    self.board[targetY-1][targetX] = None

            # If piece selected is a pawn, and it takes a piece, find which direction it took the piece (from the left or right)
            # this is important for algebraic notation.
            if (isinstance(self.board[curY][curX], Pawn) == True):
                if (targetX < curX):
                    toLeft = True
                if (targetX > curX):
                    toLeft = False

            # If the piece is a rook, pawn or a king, and we just moved it, change the hasMoved field to True
            if (isinstance(self.board[curY][curX], (Rook, King, Pawn)) == True):
                self.board[curY][curX].hasMoved = True
                # If piece is a king, then store its coordinates in the appropriate board field
                if (isinstance(self.board[curY][curX], King) == True):
                    if (self.board[curY][curX].color == True):
                        self.whiteKing = (targetY, targetX)
                        # If white king queenside castled, move rook on a1 to d1
                        if (targetX-curX == -2):
                            self.board[7][3] = self.board[7][0]
                            self.board[7][0] = None
                            self.board[7][3].hasMoved = True
                            self.board[7][3].rankNum = 7
                            self.board[7][3].fileNum = 3
                        # If white king kingside castled, move rook on h1 to f1
                        if (targetX-curX == 2):
                            self.board[7][5] = self.board[7][7]
                            self.board[7][7] = None
                            self.board[7][5].hasMoved = True
                            self.board[7][5].rankNum = 7
                            self.board[7][5].fileNum = 5
                    else:
                        self.blackKing = (targetY, targetX)
                        # If black king queenside castled, move rook on a8 to d8
                        if (targetX-curX == -2):
                            self.board[0][3] = self.board[0][0]
                            self.board[0][0] = None
                            self.board[0][3].hasMoved = True
                            self.board[0][3].rankNum = None
                            self.board[0][3].fileNum = 3
                        # If black king kingside castled, move rook on h8 to f8
                        if (targetX-curX == 2):
                            self.board[0][5] = self.board[0][7]
                            self.board[0][7] = None
                            self.board[0][5].hasMoved = True
                            self.board[0][5].rankNum = None
                            self.board[0][5].fileNum = 5

            self.lastMove = self.board[curY][curX].CoordToAlgebraic(
                targetCoords, promotion=promotion, takes=didTake, toLeft=toLeft)
            # Add the move to the move list of the game
            if (self.whiteToMove == True):
                self.totalMoves += 1
                self.listOfMoves.append(
                    str(self.totalMoves) + ". " + self.lastMove)
            else:
                self.listOfMoves[-1] = self.listOfMoves[-1] + \
                    " " + self.lastMove

            # Move the piece on the board and update its coordinate fields
            if (promotion == ""):
                self.board[targetY][targetX] = self.board[curY][curX]
            # If piece is a pawn being promoted, put the appropriate piece on the target square
            else:
                if (promotion == "Q"):
                    self.board[targetY][targetX] = Queen(
                        self.board[curY][curX].color, targetY, targetX, self)
                elif (promotion == "B"):
                    self.board[targetY][targetX] = Bishop(
                        self.board[curY][curX].color, targetY, targetX, self)
                elif (promotion == "N"):
                    self.board[targetY][targetX] = Knight(
                        self.board[curY][curX].color, targetY, targetX, self)
                elif (promotion == "R"):
                    self.board[targetY][targetX] = Rook(
                        self.board[curY][curX].color, targetY, targetX, self)

            # If a piece was taken, then update the lastCapture field to the current move number, used for detecting for 50 move draw rule
            if (isinstance(self.board[curY][curX], Pawn) == True or didTake == True):
                if (self.whiteToMove == True):
                    self.lastCapture = self.totalMoves
                else:
                    self.lastCapture = self.totalMoves + 1
            self.board[targetY][targetX].rankNum = targetY
            self.board[targetY][targetX].fileNum = targetX
            self.board[curY][curX] = None

            self.whiteToMove = not self.whiteToMove

            didCheck = False
            if (self.whiteToMove == True):
                didCheck = InCheck(self.board, self.whiteKing)
            else:
                didCheck = InCheck(self.board, self.blackKing)

            self.gameState = self.GameOver(self.whiteToMove)
            if (didCheck == True):
                if (self.gameState == 1):
                    self.listOfMoves[-1] += "#"
                    if (self.whiteToMove == True):
                        # White lost
                        self.listOfMoves.append("0-1")
                    else:
                        # Black lost
                        self.listOfMoves.append("1-0")
                else:
                    self.listOfMoves[-1] += "+"
            # Game is drawn
            if (self.gameState == -1):
                self.listOfMoves.append("1/2-1/2")
            return True
        else:
            return False

    # Returns -1 if game ends in stalemate for side (white or black) that the function is called with
    #  or drawn by insufficient material on the board, returns 1 if game ends in checkmate
    #  for the side that the function is called with, and returns 0 if the game continues.
    def GameOver(self, side):
        # 50 move rule for draws, if 50 moves have passed since last capture or pawn move game is automatically drawn
        if (self.totalMoves - self.lastCapture >= 50):
            return -1

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
        if (side == True):
            kingCoords = self.whiteKing
        else:
            kingCoords = self.blackKing

        # Count pieces and check to see whether piece can move
        canMove = False
        for row in self.board:
            for piece in row:
                if (isinstance(piece, Piece) == True):
                    if (piece.color == side and len(piece.MoveList()) != 0):
                        canMove = True
                if (isinstance(piece, Queen) == True):
                    if (piece.color == True):
                        whiteQueenCount += 1
                    else:
                        blackQueenCount += 1
                elif (isinstance(piece, Bishop) == True):
                    if (piece.color == True):
                        whiteBishopCount += 1
                    else:
                        blackBishopCount += 1
                elif (isinstance(piece, Rook) == True):
                    if (piece.color == True):
                        whiteRookCount += 1
                    else:
                        blackRookCount += 1
                elif (isinstance(piece, Knight) == True):
                    if (piece.color == True):
                        whiteKnightCount += 1
                    else:
                        blackKnightCount += 1
                elif (isinstance(piece, Pawn) == True):
                    if (piece.color == True):
                        whitePawnCount += 1
                    else:
                        blackPawnCount += 1

        if (canMove == False):
            inCheck = InCheck(self.board, kingCoords)
            # If no piece of the side passed can move and the king is in check, it is checkmate
            if (inCheck == True):
                return 1

            # If no piece of the side passed can move and the king is not in check, it is stalemate
            else:
                return -1

        # Check if potential insufficient material remains, if there are pawns, rooks, or queens on the board then the game
        # cannot be drawn by insufficient material
        insufficientMatRemaining = False
        if ((whiteQueenCount == 0 and blackQueenCount == 0) and (whiteRookCount == 0 and blackRookCount == 0) and (whitePawnCount == 0 and blackPawnCount == 0)):
            insufficientMatRemaining = True
        else:
            return 0

        if (insufficientMatRemaining == True):
            # Game is drawn if there are no bishops and there are ONLY 1 or less knights on either side in the game
            if ((whiteBishopCount == 0 and blackBishopCount == 0) and (whiteKnightCount < 2 and blackKnightCount < 2)):
                return -1
            # Game is drawn if there are no knights and there are ONLY 1 or less knights on either side in the game
            if ((whiteKnightCount == 0 and blackKnightCount == 0) and (whiteBishopCount < 2 and blackBishopCount < 2)):
                return -1
            # Game is drawn if there is only 2 knights vs a king
            if ((whiteKnightCount == 2 and blackBishopCount == 0) or (whiteBishopCount == 0 and blackKnightCount == 2)):
                return -1
            # Game is drawn if it's just 1 white knight vs only one black bishop
            if ((whiteKnightCount == 1 and whiteBishopCount == 0) and (blackBishopCount == 1 and blackKnightCount == 0)):
                return -1
            # Game is drawn if it's just one white bishop vs only one black knight
            if ((whiteBishopCount == 1 and whiteKnightCount == 0) and (blackBishopCount == 0 and blackKnightCount == 1)):
                return -1

        # If none of the previous conditions are met, the game continues
        return 0

    def ShowBoard(self):
        print(" -----------------")
        for row in range(0, 8):
            for col in range(0, 8):
                if (col == 0):
                    print("| ", end="")
                if (isinstance(self.board[row][col], Pawn) == True):
                    print("P ", end="")
                elif (isinstance(self.board[row][col], Queen) == True):
                    print("Q ", end="")
                elif (isinstance(self.board[row][col], Rook) == True):
                    print("R ", end="")
                elif (isinstance(self.board[row][col], Bishop) == True):
                    print("B ", end="")
                elif (isinstance(self.board[row][col], Knight) == True):
                    print("N ", end="")
                elif (isinstance(self.board[row][col], King) == True):
                    print("K ", end="")
                else:
                    print("  ", end="")
                if (col == 7):
                    print("|")
        print(" -----------------")
