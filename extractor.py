import chess
import torch
import chess.pgn

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
   
def encode_game(game):
    res = int(game.headers["Result"][0]) 
    board = game.board()
    bitboard_states = []
    for turn, move in enumerate(game.mainline_moves()):
        board.push(move)
        turn_color = (turn+1)%2
        bitboard_states.append(encode_board(board,turn_color))
    
    return bitboard_states, [res]*len(bitboard_states)

def get_dataset(path, num_games=50):
    games = parse(path,num_games)
    X, Y = [], []
    for idx, game in enumerate(games):
        x, y = encode_game(game)
        print(f'encoded game {idx+1}')
        X += x
        Y += y   
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y