import pygame
import ChessGame
import time
import numpy as np

FPS = 60

PIECE_IMAGES = {
    "wK" : pygame.image.load(r"ChessPieceImages\WhiteKing.png"),
    "wQ" : pygame.image.load(r"ChessPieceImages\WhiteQueen.png"),
    "wR" : pygame.image.load(r"ChessPieceImages\WhiteRook.png"),
    "wN" : pygame.image.load(r"ChessPieceImages\WhiteKnight.png"),
    "wB" : pygame.image.load(r"ChessPieceImages\WhiteBishop.png"),
    "wP" : pygame.image.load(r"ChessPieceImages\WhitePawn.png"),

    "bK" : pygame.image.load(r"ChessPieceImages\BlackKing.png"),
    "bQ" : pygame.image.load(r"ChessPieceImages\BlackQueen.png"),
    "bR" : pygame.image.load(r"ChessPieceImages\BlackRook.png"),
    "bN" : pygame.image.load(r"ChessPieceImages\BlackKnight.png"),
    "bB" : pygame.image.load(r"ChessPieceImages\BlackBishop.png"),
    "bP" : pygame.image.load(r"ChessPieceImages\BlackPawn.png")
}

def DrawBoard (screen):
    screen.fill((0,0,0))
    lightSquare = (225, 220, 190)
    darkSquare = (155, 105, 50)
    windowHeight = screen.get_height()
    squareSize = windowHeight//8
    for row in range (8):
        for col in range (8):
            if ((row + col) % 2 == 0):
                pygame.draw.rect(screen, lightSquare, pygame.Rect((col * squareSize), (row * squareSize), squareSize, squareSize))
            else:
                pygame.draw.rect(screen, darkSquare, pygame.Rect((col * squareSize), (row * squareSize), squareSize, squareSize))

def DrawPieces (board, screen, flipped):
    DrawBoard(screen)
    windowHeight = screen.get_height()
    squareSize = windowHeight//8
    if (flipped == True):
        for row in range (8):
            for col in range (8):
                if (isinstance(board.board[row][col], ChessGame.King) == True):
                    if (board.board[row][col].color == True):
                        screen.blit(pygame.transform.smoothscale(PIECE_IMAGES["wK"], (squareSize, squareSize)), pygame.Rect(abs(col-7) * squareSize, abs(row-7) * squareSize, squareSize, squareSize))
                    else:
                        screen.blit(pygame.transform.smoothscale(PIECE_IMAGES["bK"], (squareSize, squareSize)), pygame.Rect(abs(col-7) * squareSize, abs(row-7) * squareSize, squareSize, squareSize))
                elif (isinstance(board.board[row][col], ChessGame.Queen) == True):
                    if (board.board[row][col].color == True):
                        screen.blit(pygame.transform.smoothscale(PIECE_IMAGES["wQ"], (squareSize, squareSize)), pygame.Rect(abs(col-7) * squareSize, abs(row-7) * squareSize, squareSize, squareSize))
                    else:
                        screen.blit(pygame.transform.smoothscale(PIECE_IMAGES["bQ"], (squareSize, squareSize)), pygame.Rect(abs(col-7) * squareSize, abs(row-7) * squareSize, squareSize, squareSize))
                elif (isinstance(board.board[row][col], ChessGame.Rook) == True):
                    if (board.board[row][col].color == True):
                        screen.blit(pygame.transform.smoothscale(PIECE_IMAGES["wR"], (squareSize, squareSize)), pygame.Rect(abs(col-7) * squareSize, abs(row-7) * squareSize, squareSize, squareSize))
                    else:
                        screen.blit(pygame.transform.smoothscale(PIECE_IMAGES["bR"], (squareSize, squareSize)), pygame.Rect(abs(col-7) * squareSize, abs(row-7) * squareSize, squareSize, squareSize))
                elif (isinstance(board.board[row][col], ChessGame.Knight) == True):
                    if (board.board[row][col].color == True):
                        screen.blit(pygame.transform.smoothscale(PIECE_IMAGES["wN"], (squareSize, squareSize)), pygame.Rect(abs(col-7) * squareSize, abs(row-7) * squareSize, squareSize, squareSize))
                    else:
                        screen.blit(pygame.transform.smoothscale(PIECE_IMAGES["bN"], (squareSize, squareSize)), pygame.Rect(abs(col-7) * squareSize, abs(row-7) * squareSize, squareSize, squareSize))
                elif (isinstance(board.board[row][col], ChessGame.Bishop) == True):
                    if (board.board[row][col].color == True):
                        screen.blit(pygame.transform.smoothscale(PIECE_IMAGES["wB"], (squareSize, squareSize)), pygame.Rect(abs(col-7) * squareSize, abs(row-7) * squareSize, squareSize, squareSize))
                    else:
                        screen.blit(pygame.transform.smoothscale(PIECE_IMAGES["bB"], (squareSize, squareSize)), pygame.Rect(abs(col-7) * squareSize, abs(row-7) * squareSize, squareSize, squareSize))
                elif (isinstance(board.board[row][col], ChessGame.Pawn) == True):
                    if (board.board[row][col].color == True):
                        screen.blit(pygame.transform.smoothscale(PIECE_IMAGES["wP"], (squareSize, squareSize)), pygame.Rect(abs(col-7) * squareSize, abs(row-7) * squareSize, squareSize, squareSize))
                    else:
                        screen.blit(pygame.transform.smoothscale(PIECE_IMAGES["bP"], (squareSize, squareSize)), pygame.Rect(abs(col-7) * squareSize, abs(row-7) * squareSize, squareSize, squareSize))
    else:
        for row in range (8):
            for col in range (8):
                if (isinstance(board.board[row][col], ChessGame.King) == True):
                    if (board.board[row][col].color == True):
                        screen.blit(pygame.transform.smoothscale(PIECE_IMAGES["wK"], (squareSize, squareSize)), pygame.Rect(col * squareSize, row * squareSize, squareSize, squareSize))
                    else:
                        screen.blit(pygame.transform.smoothscale(PIECE_IMAGES["bK"], (squareSize, squareSize)), pygame.Rect(col * squareSize, row * squareSize, squareSize, squareSize))
                elif (isinstance(board.board[row][col], ChessGame.Queen) == True):
                    if (board.board[row][col].color == True):
                        screen.blit(pygame.transform.smoothscale(PIECE_IMAGES["wQ"], (squareSize, squareSize)), pygame.Rect(col * squareSize, row * squareSize, squareSize, squareSize))
                    else:
                        screen.blit(pygame.transform.smoothscale(PIECE_IMAGES["bQ"], (squareSize, squareSize)), pygame.Rect(col * squareSize, row * squareSize, squareSize, squareSize))
                elif (isinstance(board.board[row][col], ChessGame.Rook) == True):
                    if (board.board[row][col].color == True):
                        screen.blit(pygame.transform.smoothscale(PIECE_IMAGES["wR"], (squareSize, squareSize)), pygame.Rect(col * squareSize, row * squareSize, squareSize, squareSize))
                    else:
                        screen.blit(pygame.transform.smoothscale(PIECE_IMAGES["bR"], (squareSize, squareSize)), pygame.Rect(col * squareSize, row * squareSize, squareSize, squareSize))
                elif (isinstance(board.board[row][col], ChessGame.Knight) == True):
                    if (board.board[row][col].color == True):
                        screen.blit(pygame.transform.smoothscale(PIECE_IMAGES["wN"], (squareSize, squareSize)), pygame.Rect(col * squareSize, row * squareSize, squareSize, squareSize))
                    else:
                        screen.blit(pygame.transform.smoothscale(PIECE_IMAGES["bN"], (squareSize, squareSize)), pygame.Rect(col * squareSize, row * squareSize, squareSize, squareSize))
                elif (isinstance(board.board[row][col], ChessGame.Bishop) == True):
                    if (board.board[row][col].color == True):
                        screen.blit(pygame.transform.smoothscale(PIECE_IMAGES["wB"], (squareSize, squareSize)), pygame.Rect(col * squareSize, row * squareSize, squareSize, squareSize))
                    else:
                        screen.blit(pygame.transform.smoothscale(PIECE_IMAGES["bB"], (squareSize, squareSize)), pygame.Rect(col * squareSize, row * squareSize, squareSize, squareSize))
                elif (isinstance(board.board[row][col], ChessGame.Pawn) == True):
                    if (board.board[row][col].color == True):
                        screen.blit(pygame.transform.smoothscale(PIECE_IMAGES["wP"], (squareSize, squareSize)), pygame.Rect(col * squareSize, row * squareSize, squareSize, squareSize))
                    else:
                        screen.blit(pygame.transform.smoothscale(PIECE_IMAGES["bP"], (squareSize, squareSize)), pygame.Rect(col * squareSize, row * squareSize, squareSize, squareSize))

# FIX THIS FUNCTION LATER
""" def PossibleMoves (board, screen, flipped, pieceSelected):
    squareSize = screen.get_height() // 8
    curY, curX = pieceSelected

    if (isinstance(board.board[curY][curX], ChessGame.Piece) == True and board.board[curY][curX].color == board.whiteToMove):
        moves = board.board[curY][curX].MoveList()
        possibleCircles = pygame.Surface((screen.get_height(), screen.get_height()), pygame.SRCALPHA)
        for x in moves:
            takes = False
            targetY, targetX = x[0], x[1]
            if (isinstance(board.board[targetY][targetX], ChessGame.Piece) == True and board.board[targetY][targetX].color != board.board[curY][curX].color):
                takes = True
            if (flipped == True):
                if (takes == True):
                    pygame.draw.circle(possibleCircles, (255, 0, 0, 170), (abs(targetX-7) * squareSize + squareSize // 2, abs(targetY-7) * squareSize + squareSize //2), squareSize // 2, squareSize // 10)
                else:
                    pygame.draw.circle(possibleCircles, (160, 160, 160, 115), (abs(targetX-7) * squareSize + squareSize //2, abs(targetY-7) * squareSize + squareSize // 2), squareSize // 10)
            else:
                if (takes == True):
                    pygame.draw.circle(possibleCircles, (255, 0, 0, 170), (targetX * squareSize + squareSize // 2, targetY * squareSize + squareSize //2), squareSize // 2, squareSize // 10)
                else:
                    pygame.draw.circle(possibleCircles, (160, 160, 160, 115), (targetX * squareSize + squareSize //2, targetY * squareSize + squareSize // 2), squareSize // 10)
        screen.blit(possibleCircles, (0,0)) """


def Play (board, screen, flipped):
    clock = pygame.time.Clock()
    lastMove = []
    squareClicked = ()

    running = True
    sum = 0
    while (running == True):
        clock.tick(FPS)
        if (board.gameState != 0):
            break
        for event in pygame.event.get():
            if (event.type == pygame.QUIT):
                running = False
            elif (pygame.mouse.get_pressed()[0] == True):
                # Find the coordinates on the pygame window that the user clicked (x, y)
                col, row = pygame.mouse.get_pos()
                col = col // (screen.get_height() // 8)
                row = row // (screen.get_height() // 8)
                
                if (flipped == True):
                    col = 7 - col
                    row = 7 - row

                if (len(lastMove) == 0):
                    # If it is the first click the player makes and they choose the side to move's piece, then record the piece
                    # they selected.
                    if (isinstance(board.board[row][col], ChessGame.Piece) == True and board.board[row][col].color == board.whiteToMove):
                        squareClicked = (row, col)
                        # PossibleMoves(board, screen, flipped, squareClicked)
                        lastMove.append((row,col, ""))
                else:
                    # If player clicks the same square, reset their moves
                    if (squareClicked == (row, col)):
                        squareClicked = ()
                        lastMove.clear()
                        # Get rid of possible move circles from last piece selected
                        DrawPieces(board, screen, flipped)
                    if (isinstance(board.board[row][col], ChessGame.Piece) == True and board.board[row][col].color == board.whiteToMove):
                        lastMove.clear()
                        squareClicked = (row, col)
                        lastMove.append(squareClicked)
                        DrawPieces(board, screen, flipped)
                        # PossibleMoves(board, screen, flipped, squareClicked)
                    # If player clicks a square after selecting a piece, try to move the selected piece to that square
                    else:
                        squareClicked = (row, col)
                        piece = board.board[lastMove[0][0]][lastMove[0][1]]
                        promotion = ""
                        # Check if piece is pawn that wants to be promoted
                        if (isinstance (piece, ChessGame.Pawn) == True):
                            possiblePromotions = pygame.Surface((screen.get_height(), screen.get_height()), pygame.SRCALPHA)
                            squareSize = screen.get_height() // 8
                            # If piece is white pawn and wants to move to the 8th rank
                            if (piece.color == True and squareClicked[0] == 0):
                                for dx in range (4):
                                    fileNum = squareClicked[1]
                                    rankNum = dx
                                    if (flipped == True):
                                        fileNum = abs(fileNum-7)
                                        rankNum = abs(rankNum-7)
                                    pygame.draw.circle(screen, (160, 160, 160, 0), (fileNum * squareSize + squareSize // 2, rankNum * squareSize + squareSize //2), squareSize // 2)
                                    if (dx == 0):
                                        possiblePromotions.blit(pygame.transform.smoothscale(PIECE_IMAGES["wQ"], (squareSize, squareSize)), pygame.Rect(fileNum * squareSize, rankNum * squareSize, squareSize, squareSize))
                                    elif (dx == 1):
                                        possiblePromotions.blit(pygame.transform.smoothscale(PIECE_IMAGES["wN"], (squareSize, squareSize)), pygame.Rect(fileNum * squareSize, rankNum * squareSize, squareSize, squareSize))
                                    elif (dx == 2):
                                        possiblePromotions.blit(pygame.transform.smoothscale(PIECE_IMAGES["wR"], (squareSize, squareSize)), pygame.Rect(fileNum * squareSize, rankNum * squareSize, squareSize, squareSize))
                                    elif (dx == 3):
                                        possiblePromotions.blit(pygame.transform.smoothscale(PIECE_IMAGES["wB"], (squareSize, squareSize)), pygame.Rect(fileNum * squareSize, rankNum * squareSize, squareSize, squareSize))
                                screen.blit (possiblePromotions, (0,0))
                                pygame.display.update()
                                gotInput = False
                                reset = False
                                while (gotInput == False):
                                    for event in pygame.event.get():
                                        if (event.type == pygame.QUIT):
                                            reset = True
                                            running = False
                                        if (pygame.mouse.get_pressed()[0] == True):
                                            gotInput = True
                                        elif (pygame.mouse.get_pressed()[2] == True):
                                            squareClicked = ()
                                            lastMove.clear()
                                            # Get rid of possible move circles from last piece selected
                                            DrawPieces(board, screen, flipped)
                                            pygame.display.update()
                                            reset = True
                                            gotInput = True
                                if (reset == True):
                                    continue
                            if (piece.color == False and squareClicked[0] == 7):
                                for dx in range (4, 8):
                                    fileNum = squareClicked[1]
                                    rankNum = dx
                                    if (flipped == True):
                                        fileNum = abs(fileNum-7)
                                        rankNum = abs(rankNum-7)
                                    pygame.draw.circle(screen, (160, 160, 160, 0), (fileNum * squareSize + squareSize // 2, rankNum * squareSize + squareSize //2), squareSize // 2)
                                    if (dx == 7):
                                        possiblePromotions.blit(pygame.transform.smoothscale(PIECE_IMAGES["bQ"], (squareSize, squareSize)), pygame.Rect(fileNum * squareSize, rankNum * squareSize, squareSize, squareSize))
                                    elif (dx == 6):
                                        possiblePromotions.blit(pygame.transform.smoothscale(PIECE_IMAGES["bN"], (squareSize, squareSize)), pygame.Rect(fileNum * squareSize, rankNum * squareSize, squareSize, squareSize))
                                    elif (dx == 5):
                                        possiblePromotions.blit(pygame.transform.smoothscale(PIECE_IMAGES["bR"], (squareSize, squareSize)), pygame.Rect(fileNum * squareSize, rankNum * squareSize, squareSize, squareSize))
                                    elif (dx == 4):
                                        possiblePromotions.blit(pygame.transform.smoothscale(PIECE_IMAGES["bB"], (squareSize, squareSize)), pygame.Rect(fileNum * squareSize, rankNum * squareSize, squareSize, squareSize))
                                screen.blit (possiblePromotions, (0,0))
                                pygame.display.update()
                                gotInput = False
                                reset = False
                                while (gotInput == False):
                                    for event in pygame.event.get():
                                        if (event.type == pygame.QUIT):
                                            reset = True
                                            running = False
                                        if (pygame.mouse.get_pressed()[0] == True):
                                            gotInput = True
                                        elif (pygame.mouse.get_pressed()[2] == True):
                                            squareClicked = ()
                                            lastMove.clear()
                                            # Get rid of possible move circles from last piece selected
                                            DrawPieces(board, screen, flipped)
                                            pygame.display.update()
                                            reset = True
                                            gotInput = True
                                if (reset == True):
                                    continue
                            
                            # Find the coordinates on the pygame window that the user clicked (x, y)
                            promoCol, promoRow = pygame.mouse.get_pos()
                            promoCol = promoCol // (screen.get_height() // 8)
                            promoRow = promoRow // (screen.get_height() // 8)
                            if (flipped == True):
                                promoCol = abs(col-7)
                                promoRow = abs(row-7)
                                    
                            if (squareClicked[0] == 0):
                                if (promoRow == 0):
                                    promotion = ""
                                if (promoRow == 1):
                                    promotion = "N"
                                if (promoRow == 2):
                                    promotion = "R"
                                if (promoRow == 3):
                                    promotion = "B"
                            elif (squareClicked[0] == 7):
                                if (promoRow == 7):
                                    promotion = ""
                                if (promoRow == 6):
                                    promotion = "N"
                                if (promoRow == 5):
                                    promotion = "R"
                                if (promoRow == 4):
                                    promotion = "B"

                        # If the move is legal, the Move function will change the board appropriately and change whose move it is
                        # If not, the move is not made and the moves are reset and it's still the same player's move
                        squareClicked = (row, col, promotion)
                        lastMove.append(squareClicked)
                        
                        
                        dy = squareClicked[0]-lastMove[0][0]
                        dx = squareClicked[1]-lastMove[0][1]

                        rankNum = lastMove[0][0]
                        fileNum = lastMove[0][1]

                        if (board.whiteToMove == False):
                            rankNum = 7 - rankNum
                            fileNum = 7 - fileNum
                        
                        moveCode = board.EncodeMove((dy, dx), promotion)

                        # If moveCode is None, it means potential move is completely illegal (i.e. does not even exist in the game of chess),
                        # so clear the move queue and the last squareClicked
                        if (moveCode == None):
                            squareClicked = ()
                            lastMove.clear()
                            # Get rid of possible move circles from last piece selected
                            DrawPieces(board, screen, flipped)
                            pygame.display.update()
                            continue
                            
                        moveTuple = (rankNum, fileNum, moveCode)
                        start_time = time.time()
                        board.Move(np.ravel_multi_index((moveTuple), (8,8,73)))
                        evaluation = ChessGame.Eval(board, color = board.whiteToMove)
                        end_time = time.time()
                        
                        print (evaluation)
                        print ("Calculated in " + str(end_time-start_time) + " seconds.")
                        squareClicked = ()
                        lastMove.clear()
                        DrawPieces(board, screen, flipped)
            elif (pygame.mouse.get_pressed()[2] == True):
                squareClicked = ()
                lastMove.clear()
                # Get rid of possible move circles from last piece selected
                DrawPieces(board, screen, flipped)
        pygame.display.update()

    # If checkmate, print the appropriate victor
    if (board.gameState == 1):
        # If it's checkmate and it's white's move, white lost
        if (board.whiteToMove == True):
            print ("BLACK WINS BY CHECKMATE!")
        # If it's checkmate and it's black's move, black lost
        else:
            print ("WHITE WINS BY CHECKMATE!")

    # If stalemate, print message stating game is drawn
    if (board.gameState == -1):
        print ("GAME IS DRAWN.")
    

def main ():
    screen = pygame.display.set_mode((640,640), pygame.RESIZABLE, pygame.SRCALPHA)
    pygame.display.set_caption("Chess Game: ")

    board = ChessGame.Board()

    # board.board[0][0] = ChessGame.King(True, 0, 0, board)
    # board.board[7][7] = ChessGame.King(False, 7, 7, board)
    # board.board[6][5] = ChessGame.Rook(False, 6, 5, board)
    # board.board[6][6] = ChessGame.Pawn(False, 6, 6, board)

    flipped = False
    DrawPieces (board, screen, flipped)
    Play(board, screen, flipped)

    for x in board.listOfMoves:
        print (x + " ", end = "")


if __name__ == "__main__":
    main()