#!/bin/python3

import sys
import chess
import numpy as np 
import chess.pgn
import random 

def parse(path, limit):
    pgn = open(path)
    results=[]
    X, Y = [], []
    c = 0
    white_games, black_games = 0, 0
    while(c < limit):
        g = chess.pgn.read_game(pgn)
        res = int(g.headers["Result"][0]) 
        if white_games == limit//2 and res == 1:
            continue
        if g is None: break
        if g.headers["Result"] != "1/2-1/2" : 
            x, y = encode_game(g)#ignore games that end in a draw 
            X += x
            Y += y
            c += 1
            white_games += res
            black_games += int(not(res))

        print('games parsed %d'%(c))

    print('done parsing %d games'%(c))

    pgn.close()
    return(X,Y)

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
        bitboard_states.append(encode_board(board,turn_color))
    # random.shuffle(bitboard_states)
    # return bitboard_states[:15], [res]*15
    return bitboard_states, [res]*len(bitboard_states)

def prepare_dataset(name, num_games=50):
    path = 'dataset/games.pgn'
    X, Y = parse(path,num_games)
    print('saving to disk as numpy arrays')
    X = np.array(X)
    Y = np.array(Y)
    np.savez(name, X, Y)
    print(X.shape, Y.shape)

if __name__ == "__main__":
    name = 'data20k_x.npz'
    num_games = 39880
    prepare_dataset(name,num_games)
