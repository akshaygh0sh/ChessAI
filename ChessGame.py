import time

# Chess Board:
# Black side
# ---------------
# R N B Q K B N R 
# P P P P P P P P




# P P P P P P P P
# R N B Q K B N R
# ---------------
# White side

# Determines if a square is in Check or not given the board it is placed on and its coordinates
def InCheck (board, coordinates):
        rankNum, fileNum = coordinates

        # Check for pawn checks on White King
        if (board[rankNum][fileNum] != 0 and board[rankNum][fileNum].color == True):
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
        elif (board[rankNum][fileNum] != 0 and board[rankNum][fileNum].color == False):
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
            if ((targetX >= 0 and targetX  <= 7) and (targetY >= 0 and targetY <= 7)):
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
        for y in range (rankNum -1, -1, -1):
            if (board[y][fileNum] != 0 and board[y][fileNum].color != board[rankNum][fileNum].color):
                if (isinstance(board[y][fileNum], Rook) == True):
                    return True
                else:
                    break
            elif (board[y][fileNum] != 0 and board[y][fileNum].color == board[rankNum][fileNum].color):
                break
        
        for y in range (rankNum + 1, 8):
            if (board[y][fileNum] != 0 and board[y][fileNum].color != board[rankNum][fileNum].color):
                if (isinstance(board[y][fileNum], Rook) == True):
                    return True
                else:
                    break
            elif (board[y][fileNum] != 0 and board[y][fileNum].color == board[rankNum][fileNum].color):
                break
        
        # Check for horizontal checks from rooks/queens
        for x in range (fileNum-1, -1, -1):
            if (board[rankNum][x] != 0 and board[rankNum][x].color != board[rankNum][fileNum].color):
                if (isinstance(board[rankNum][x], Rook) == True):
                    return True
                else:
                    break
            elif (board[rankNum][x] != 0 and board[rankNum][x].color == board[rankNum][fileNum].color):
                break
        
        for x in range (fileNum + 1, 8):
            if (board[rankNum][x] != 0 and board[rankNum][x].color != board[rankNum][fileNum].color):
                if (isinstance(board[rankNum][x], Rook) == True):
                    return True
                else:
                    break
            elif (board[rankNum][x] != 0 and board[rankNum][x].color == board[rankNum][fileNum].color):
                break

        # Check for diagonal checks from bishops/queens
        x = fileNum
        y = rankNum
        # Check north west diagonal checks
        while (x > 0 and y > 0):
            if (board[y-1][x-1] == 0):
                y-=1
                x-=1
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
            if (board[y-1][x+1] == 0):
                y-=1
                x+=1
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
            if (board[y+1][x-1] == 0):
                y+=1
                x-=1
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
            if (board[y+1][x+1] == 0):
                y+=1
                x+=1
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
    def __init__ (self, color, rankNum, fileNum, board):
        self.color = color
        self.rankNum = rankNum
        self.fileNum = fileNum
        self.board = board

    def CoordToAlgebraic (self, coordinate, promotion = "", takes = False, toLeft = False):
        rankNum, fileNum = coordinate
        
        # If king has castled
        if (isinstance(self, King) == True):
            # Queenside castle
            if (fileNum-self.fileNum == -2):
                return "0-0-0"
            elif (fileNum-self.fileNum == 2):
                return "0-0"
        
        files = {
            0 : "a",
            1 : "b",
            2 : "c", 
            3 : "d", 
            4 : "e",
            5 : "f",
            6 : "g",
            7 : "h"
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
    def MoveList (self):
        moves = set()
        kingCoord = ()
        if (self.color == True):
            kingCoord = self.board.whiteKing
        else:
            kingCoord = self.board.blackKing

        # If pawn is white
        if (self.color == True):
            # If square one in front is empty, pawn can move there
            if (self.board.board[self.rankNum-1][self.fileNum] == 0):
                # Change board as if the potential move was played
                self.board.board[self.rankNum-1][self.fileNum] = self
                self.board.board[self.rankNum][self.fileNum] = 0

                # Check to see if respective side's king is in check after the move
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    moves.add((self.rankNum-1, self.fileNum))
                
                # Change board back to original state
                self.board.board[self.rankNum-1][self.fileNum] = 0
                self.board.board[self.rankNum][self.fileNum] = self

            # If diagonal pawn capture squares contain black pieces, pawn can move there
            if (self.fileNum - 1 >= 0 and self.board.board[self.rankNum-1][self.fileNum-1] != 0 and self.board.board[self.rankNum-1][self.fileNum-1].color != self.color):
                piece = self.board.board[self.rankNum-1][self.fileNum-1]

                # Change board as if the potential move was played
                self.board.board[self.rankNum-1][self.fileNum-1] = self
                self.board.board[self.rankNum][self.fileNum] = 0

                # Check to see if respective side's king is in check after the move
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    moves.add((self.rankNum-1, self.fileNum-1))
                
                # Change board back to original state
                self.board.board[self.rankNum-1][self.fileNum-1] = piece
                self.board.board[self.rankNum][self.fileNum] = self

            if (self.fileNum + 1 <= 7 and self.board.board[self.rankNum-1][self.fileNum+1] != 0 and self.board.board[self.rankNum-1][self.fileNum+1].color != self.color):
                piece = self.board.board[self.rankNum-1][self.fileNum+1]

                # Change board as if the potential move was played
                self.board.board[self.rankNum-1][self.fileNum+1] = self
                self.board.board[self.rankNum][self.fileNum] = 0

                # Check to see if respective side's king is in check after the move
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    moves.add((self.rankNum-1, self.fileNum+1))
                
                # Change board back to original state
                self.board.board[self.rankNum-1][self.fileNum+1] = piece
                self.board.board[self.rankNum][self.fileNum] = self
        # If pawn is black
        else:
            # If square one in front is empty, pawn can move there
            if (self.board.board[self.rankNum+1][self.fileNum] == 0):
                # Change board as if the potential move was played
                self.board.board[self.rankNum+1][self.fileNum] = self
                self.board.board[self.rankNum][self.fileNum] = 0

                # Check to see if respective side's king is in check after the move
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    moves.add((self.rankNum+1, self.fileNum))

                # Change board back to original state
                self.board.board[self.rankNum+1][self.fileNum] = 0
                self.board.board[self.rankNum][self.fileNum] = self
            # If diagonal pawn capture squares contain black pieces, pawn can move there
            if (self.fileNum - 1 >= 0 and self.board.board[self.rankNum+1][self.fileNum-1] != 0 and self.board.board[self.rankNum+1][self.fileNum-1].color != self.color):
                piece = self.board.board[self.rankNum+1][self.fileNum-1]

                # Change board as if the potential move was played
                self.board.board[self.rankNum+1][self.fileNum-1] = self
                self.board.board[self.rankNum][self.fileNum] = 0
                
                # Check to see if respective side's king is in check after the move
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    moves.add((self.rankNum+1, self.fileNum-1))
                
                # Change board back to original state
                self.board.board[self.rankNum+1][self.fileNum-1] = piece
                self.board.board[self.rankNum][self.fileNum] = self

            if (self.fileNum + 1 <= 7 and self.board.board[self.rankNum+1][self.fileNum+1] != 0 and self.board.board[self.rankNum+1][self.fileNum+1].color != self.color):
                piece = self.board.board[self.rankNum+1][self.fileNum+1]

                # Change board as if the potential move was played
                self.board.board[self.rankNum+1][self.fileNum+1] = self
                self.board.board[self.rankNum][self.fileNum] = 0

                # Check to see if respective side's king is in check after the move
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    moves.add((self.rankNum+1, self.fileNum+1))
                
                # Change board back to original state
                self.board.board[self.rankNum+1][self.fileNum+1] = piece
                self.board.board[self.rankNum][self.fileNum] = self
        # If pawn hasn't moved
        if (self.hasMoved == False):
            # If pawn is white can move up the board 2 squares
            if (self.color == True):
                if (self.board.board[4][self.fileNum] == 0 and self.board.board[5][self.fileNum] == 0):
                    # Change board as if the potential move was played
                    self.board.board[4][self.fileNum] = self
                    self.board.board[self.rankNum][self.fileNum] = 0
                    
                    # Check to see if respective side's king is in check after the move
                    inCheck = InCheck(self.board.board, (kingCoord))
                    if (inCheck != True):
                        moves.add((4, self.fileNum))
                    
                    # Change board back to original state
                    self.board.board[4][self.fileNum] = 0
                    self.board.board[self.rankNum][self.fileNum] = self
            # If pawn is black can move down the board 2 squares
            else:
                if (self.board.board[3][self.fileNum] == 0 and self.board.board[2][self.fileNum] == 0):
                    # Change board as if the potential move was played
                    self.board.board[3][self.fileNum] = self
                    self.board.board[self.rankNum][self.fileNum] = 0

                    # Check to see if respective side's king is in check after the move
                    inCheck = InCheck(self.board.board, (kingCoord ))
                    if (inCheck != True):
                        moves.add((3, self.fileNum))
                    
                    # Change board back to original state
                    self.board.board[3][self.fileNum] = 0
                    self.board.board[self.rankNum][self.fileNum] = self
        self.legalMoves = moves
        return moves

                    

class King(Piece):
    hasMoved = False

    def MoveList (self):
        moves = set()
        # Copy of board to test for checks
        for dy, dx in ((1, 1), (1, -1), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 0), (-1, 1)):
            targetY = self.rankNum + dy
            targetX = self.fileNum + dx
            if ((targetX >= 0 and targetX  <= 7) and (targetY >= 0 and targetY <= 7)):
                if (self.board.board[targetY][targetX] != 0 and self.board.board[targetY][targetX].color == self.color):
                    continue
                else:
                    # Change board as if potential move had been played
                    piece = self.board.board[targetY][targetX]
                    self.board.board[targetY][targetX] = self
                    self.board.board[self.rankNum][self.fileNum] = 0
                    
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
                if (self.board.board[7][5] == 0 and self.board.board[7][6] == 0):
                    canCastle = True
                    for x in (5, 6):
                        self.board.board[7][x] = self
                        self.board.board[7][4] = 0

                        canCastle = not InCheck(self.board.board, (7, x))

                        self.board.board[7][x] = 0
                        self.board.board[7][4] = self

                        # If the square is attacked, then king can't castle so break
                        if (canCastle == False):
                            break

                    if (canCastle == True):
                        moves.add((7, 6))
            # Check queenside castle
            if (isinstance(self.board.board[7][0], Rook) == True and self.board.board[7][0].hasMoved == False):
                if (self.board.board[7][3] == 0 and self.board.board[7][2] == 0 and self.board.board[7][1] == 0):
                    canCastle = True
                    canCastle = True
                    for x in (3, 2):
                        self.board.board[7][x] = self
                        self.board.board[7][4] = 0

                        canCastle = not InCheck(self.board.board, (7, x))

                        self.board.board[7][x] = 0
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
                if (self.board.board[0][5] == 0 and self.board.board[0][6] == 0):
                    canCastle = True
                    for x in (5, 6):
                        self.board.board[0][x] = self
                        self.board.board[0][4] = 0

                        canCastle = not InCheck(self.board.board, (0, x))

                        self.board.board[0][x] = 0
                        self.board.board[0][4] = self

                        # If the square is attacked, then king can't castle so break
                        if (canCastle == False):
                            break

                    if (canCastle == True):
                        moves.add((0, 6))
            # Check queenside castle
            if (isinstance(self.board.board[0][0], Rook) == True and self.board.board[0][0].hasMoved == False):
                if (self.board.board[0][3] == 0 and self.board.board[0][2] == 0 and self.board.board[0][1] == 0):
                    canCastle = True
                    for x in (3, 2):
                        self.board.board[0][x] = self
                        self.board.board[0][4] = 0

                        canCastle = not InCheck(self.board.board, (0, x))

                        self.board.board[0][x] = 0
                        self.board.board[0][4] = self

                        # If the square is attacked, then king can't castle so break
                        if (canCastle == False):
                            break

                    if (canCastle == True):
                        moves.add((0, 2))
        
        self.legalMoves = moves
        return moves


class Bishop(Piece):
    def MoveList (self):
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
            if (self.board.board[y-1][x-1] == 0):
                # Change board as if potential move had been played
                self.board.board[y-1][x-1] = self
                self.board.board[self.rankNum][self.fileNum] = 0

                # Check to see if respective king is in check
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    moves.add((y-1, x-1))
                
                # Change board back to original state
                self.board.board[y-1][x-1] = 0
                self.board.board[self.rankNum][self.fileNum] = self
                y-=1
                x-=1
            # If the potential square has a piece of the opposite color, bishop can take that piece but go no further
            # on the diagonal so add that move to the legalMoves and then break
            elif (self.board.board[y-1][x-1].color != self.color):
                piece = self.board.board[y-1][x-1]
                # Change board as if potential move had been played
                self.board.board[y-1][x-1] = self
                self.board.board[self.rankNum][self.fileNum] = 0

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
            if (self.board.board[y-1][x+1] == 0):
                # Change board as if potential move had been played
                self.board.board[y-1][x+1] = self
                self.board.board[self.rankNum][self.fileNum] = 0

                # Check to see if respective king is in check
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    moves.add((y-1, x+1))
                
                # Change board back to original state
                self.board.board[y-1][x+1] = 0
                self.board.board[self.rankNum][self.fileNum] = self
                y-=1
                x+=1
            elif (self.board.board[y-1][x+1].color != self.color):
                piece = self.board.board[y-1][x+1]
                # Change board as if potential move had been played
                self.board.board[y-1][x+1] = self
                self.board.board[self.rankNum][self.fileNum] = 0

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
            if (self.board.board[y+1][x-1] == 0):
                # Change board as if potential move had been played
                self.board.board[y+1][x-1] = self
                self.board.board[self.rankNum][self.fileNum] = 0
                
                # Check to see if respective king is in check
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    moves.add((y+1, x-1))
                
                # Change board back to original state
                self.board.board[y+1][x-1] = 0
                self.board.board[self.rankNum][self.fileNum] = self
                y+=1
                x-=1
            elif (self.board.board[y+1][x-1].color != self.color):
                piece = self.board.board[y+1][x-1]
                # Change board as if potential move had been played
                self.board.board[y+1][x-1] = self
                self.board.board[self.rankNum][self.fileNum] = 0

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
            if (self.board.board[y+1][x+1] == 0):
                # Change board as if potential move had been played
                self.board.board[y+1][x+1] = self
                self.board.board[self.rankNum][self.fileNum] = 0
                
                # Check to see if respective king is in check
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    moves.add((y+1, x+1))
                
                # Change board back to original state
                self.board.board[y+1][x+1] = 0
                self.board.board[self.rankNum][self.fileNum] = self
                y+=1
                x+=1
                
            elif (self.board.board[y+1][x+1].color != self.color):
                piece = self.board.board[y+1][x+1]
                # Change board as if potential move had been played
                self.board.board[y+1][x+1] = self
                self.board.board[self.rankNum][self.fileNum] = 0

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
    def MoveList (self):
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
                if (self.board.board[newRank][newFile] == 0):
                    
                    self.board.board[newRank][newFile] = self
                    self.board.board[self.rankNum][self.fileNum] = 0

                    inCheck = InCheck(self.board.board, (kingCoord))
                    if (inCheck != True):
                        moves.add((newRank, newFile))
                    
                    self.board.board[newRank][newFile] = 0
                    self.board.board[self.rankNum][self.fileNum] = self
                # If an enemy piece is on the potential move's square, the move is legal and the knight
                # can take the piece
                elif (self.board.board[newRank][newFile].color != self.color):
                    piece = self.board.board[newRank][newFile]

                    self.board.board[newRank][newFile] = self
                    self.board.board[self.rankNum][self.fileNum] = 0
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
    
    def MoveList (self):
        moves = set()
        kingCoord = ()
        if (self.color == True):
            kingCoord = self.board.whiteKing
        else:
            kingCoord = self.board.blackKing
        
        # Check all legal vertical moves from rook's current position to the left
        for y in range (self.rankNum -1, -1, -1):
            # If potential position is empty, rook can move there
            if (self.board.board[y][self.fileNum] == 0):
                # Change board as if potential move had been played
                self.board.board[y][self.fileNum] = self
                self.board.board[self.rankNum][self.fileNum] = 0

                # Check to see if respective side king is in check after the move has been played
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    moves.add((y, self.fileNum))
                
                # Change board back to original state
                self.board.board[y][self.fileNum] = 0
                self.board.board[self.rankNum][self.fileNum] = self
            # If potential position has a piece of the opposite color, rook can take that piece but go no further, so add that move
            # to the legalMoves and then break
            elif (self.board.board[y][self.fileNum].color != self.color):
                piece = self.board.board[y][self.fileNum]
                # Change board as if potential move had been played
                self.board.board[y][self.fileNum] = self
                self.board.board[self.rankNum][self.fileNum] = 0

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
        for y in range (self.rankNum + 1, 8):
            if (self.board.board[y][self.fileNum] == 0):
                # Change board as if potential move had been played
                self.board.board[y][self.fileNum] = self
                self.board.board[self.rankNum][self.fileNum] = 0

                # Check to see if respective side king is in check after the move has been played
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    moves.add((y, self.fileNum))
                
                # Change board back to original state
                self.board.board[y][self.fileNum] = 0
                self.board.board[self.rankNum][self.fileNum] = self
            elif (self.board.board[y][self.fileNum].color != self.color):
                piece = self.board.board[y][self.fileNum] 
                # Change board as if potential move had been played
                self.board.board[y][self.fileNum] = self
                self.board.board[self.rankNum][self.fileNum] = 0
                
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
        for x in range (self.fileNum -1, -1, -1):
            if (self.board.board[self.rankNum][x] == 0):
                # Change board as if potential move had been played
                self.board.board[self.rankNum][x] = self
                self.board.board[self.rankNum][self.fileNum] = 0

                # Check to see if respective side king is in check after the move has been played
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    moves.add((self.rankNum, x))
                
                # Change board back to original state
                self.board.board[self.rankNum][x] = 0
                self.board.board[self.rankNum][self.fileNum] = self

            elif (self.board.board[self.rankNum][x].color != self.color):
                piece = self.board.board[self.rankNum][x]
                # Change board as if potential move had been played
                self.board.board[self.rankNum][x] = self
                self.board.board[self.rankNum][self.fileNum] = 0

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
        for x in range (self.fileNum + 1, 8):
            if (self.board.board[self.rankNum][x] == 0):
                # Change board as if potential move had been played
                self.board.board[self.rankNum][x] = self
                self.board.board[self.rankNum][self.fileNum] = 0

                # Check to see if respective side king is in check after the move has been played
                inCheck = InCheck(self.board.board, (kingCoord))
                if (inCheck != True):
                    moves.add((self.rankNum, x))
                
                # Change board back to original state
                self.board.board[self.rankNum][x] = 0
                self.board.board[self.rankNum][self.fileNum] = self
            elif (self.board.board[self.rankNum][x].color != self.color):
                piece = self.board.board[self.rankNum][x]
                # Change board as if potential move had been played
                self.board.board[self.rankNum][x] = self
                self.board.board[self.rankNum][self.fileNum] = 0
                
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
    def MoveList (self):
        # Queen moves are just a combination of the set of possible rook moves and possible bishop moves.
        self.legalMoves.clear()
        self.legalMoves = Bishop.MoveList(self) | Rook.MoveList(self)
        return self.legalMoves

class Board ():
    blackKing = (())
    whiteKing = (())
    listOfMoves = []
    lastMove = ""
    totalMoves = 0

    def __init__ (self):
        board = []
        # Initalize board to 8 by 8
        for x in range (8):
            board.append([0] * 8)
        self.board = board

    def InitalizeBoard (self):
        # Set the position of the pieces
        for x in range (0, 8):
            # Set pawn positions
            self.board[1][x] = Pawn (False, 1, x, self)
            self.board[6][x] = Pawn (True, 6, x, self)

            # Set rook positions
            if (x == 0 or x == 7):
                self.board[0][x] = Rook (False, 0, x, self)
                self.board[7][x] = Rook (True, 7, x, self)   

            # Set bishop positions
            if (x == 2 or x == 5):
                self.board[0][x] = Bishop (False, 0, x, self)
                self.board[7][x] = Bishop (True, 7, x, self)   
            
            # Set knight positions
            if (x == 1 or x == 6):
                self.board[0][x] = Knight (False, 0, x, self)
                self.board[7][x] = Knight (True, 7, x, self) 

            # Set queen positions
            if (x == 3):
                self.board[0][x] = Queen (False, 0, x, self)
                self.board[7][x] = Queen (True, 7, x, self)
            
            # Set king positions
            if (x == 4):
                self.board[0][x] = King (False, 0, x, self)
                self.blackKing = ((0, x))
                self.board[7][x] = King (True, 7, x, self)
                self.whiteKing = ((7, x))

    # Moves piece on the board, returns true if move was completed (and consequently legal) and false if move was not able
    # to be completed (move was illegal or the coordinates provided did not contain a piece)
    def Move (self, curCoords, targetCoords):
        curY, curX = curCoords
        targetY, targetX = targetCoords
        
        moves = self.board[curY][curX].MoveList()
        # print (moves)
        if (targetCoords in moves):
            didTake = False
            toLeft = False
            # If a piece was taken, set the didTake variable to True (used for algebraic notation of the game)
            if (self.board[targetY][targetX] != 0):
                if (self.board[targetY][targetX].color != self.board[curY][curX].color):
                    didTake = True
            
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
                            self.board[7][0] = 0
                            self.board[7][3].hasMoved = True
                            self.board[7][3].rankNum = 7
                            self.board[7][3].fileNum = 3
                        # If white king kingside castled, move rook on h1 to f1
                        if (targetX-curX == 2):
                            self.board[7][5] = self.board[7][7]
                            self.board[7][7] = 0
                            self.board[7][5].hasMoved = True
                            self.board[7][5].rankNum = 7
                            self.board[7][5].fileNum = 5
                    else:
                        self.blackKing = (targetY, targetX)
                        # If black king queenside castled, move rook on a8 to d8
                        if (targetX-curX == -2):
                            self.board[0][3] = self.board[0][0]
                            self.board[0][0] = 0
                            self.board[0][3].hasMoved = True
                            self.board[0][3].rankNum = 0
                            self.board[0][3].fileNum = 3
                        # If black king kingside castled, move rook on h8 to f8
                        if (targetX-curX == 2):
                            self.board[0][5] = self.board[0][7]
                            self.board[0][7] = 0
                            self.board[0][5].hasMoved = True
                            self.board[0][5].rankNum = 0
                            self.board[0][5].fileNum = 5

            self.lastMove = self.board[curY][curX].CoordToAlgebraic(targetCoords, takes = didTake, toLeft = toLeft)
            # Add the move to the move list of the game
            if (self.totalMoves % 2 == 0):
                self.listOfMoves.append(str(self.totalMoves // 2 + 1) + ". " + self.lastMove)
            else:
                self.listOfMoves[-1] = self.listOfMoves[-1] + " " + self.lastMove

            # Move the piece on the board and update its coordinate fields
            self.board[targetY][targetX] = self.board[curY][curX]
            self.board[targetY][targetX].rankNum = targetY
            self.board[targetY][targetX].fileNum = targetX
            self.board[curY][curX] = 0
            self.totalMoves+=1
            return True
        else:
            return False


    def ShowBoard (self):
        print (" -----------------")
        for row in range (0, 8):
            for col in range (0, 8):
                if (col == 0):
                    print ("| ", end = "")
                if (isinstance(self.board[row][col], Pawn) == True):
                    print ("P ", end = "")
                elif (isinstance(self.board[row][col], Queen) == True):
                    print ("Q ", end = "")
                elif (isinstance(self.board[row][col], Rook) == True):
                    print ("R ", end = "")
                elif (isinstance(self.board[row][col], Bishop) == True):
                    print ("B ", end = "")
                elif (isinstance(self.board[row][col], Knight) == True):
                    print ("N ", end = "")
                elif (isinstance(self.board[row][col], King) == True):
                    print ("K ", end = "")
                else:
                    print ("  ", end = "")
                if (col == 7):
                    print ("|")
        print (" -----------------")