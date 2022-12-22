import chess
import torch
import chess.pgn
import random

def parse(path, limit):
    pgn = open(path)
    games=[]
    results=[]
    c = 0
    while(c < limit):
        g = chess.pgn.read_game(pgn)
        if g is None: break
        c+=1
        if g.headers["Result"] != "1/2-1/2" : games.append(g) #ignore games that end in a draw 
        print('games parsed %d'%(c))

    print('done parsing %d games'%(c))

    print(f"games loaded = {len(games)}")
    pgn.close()
    return(games)

def encode_board(board,move_color):
    bitboard = []
    for color in chess.COLORS:
        for p in chess.PIECE_TYPES:
            bitboard += board.pieces(p,color).tolist()
    bitboard.append(int(move_color))
    bitboard.append(board.has_kingside_castling_rights(chess.WHITE))
    bitboard.append(board.has_queenside_castling_rights(chess.WHITE))
    bitboard.append(board.has_kingside_castling_rights(chess.BLACK))
    bitboard.append(board.has_queenside_castling_rights(chess.BLACK))
    return bitboard 
   
def encode_game(game): #randomly extract 10 positions excluding the first 5 moves and captures
    res = int(game.headers["Result"][0]) 
    board = game.board()
    bitboard_states = []
    for turn, move in enumerate(game.mainline_moves()):
        capture_move = board.is_capture(move)
        board.push(move)
        turn_color = (turn+1)%2
        if turn > 5 and not capture_move:
            bitboard_states.append(encode_board(board,turn_color))
    random.shuffle(bitboard_states)
    return bitboard_states[:10], [res]*10

def get_dataset(path, num_games=50, device='cpu'):
    games = parse(path,num_games)
    X, Y = [], []
    for idx, game in enumerate(games):
        x, y = encode_game(game)
        print(f'encoded game {idx+1}')
        X += x
        Y += y   
    X = torch.tensor(X,dtype=torch.float,device=device)
    Y = torch.tensor(Y,dtype=torch.float,device=device)
    print('converting dataset into tensors, device = {device}')
    return X, Y