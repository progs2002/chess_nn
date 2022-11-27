from parser import parse 
import chess
import torch

def encode(board,move_color):
    bitboard = []
    for color in chess.COLORS:
        for p in chess.PIECE_TYPES:
            bitboard += board.pieces(p,color).tolist()
    bitboard.append(int(move_color))
    bitboard.append(board.has_kingside_castling_rights(chess.WHITE))
    bitboard.append(board.has_queenside_castling_rights(chess.WHITE))
    bitboard.append(board.has_kingside_castling_rights(chess.BLACK))
    bitboard.append(board.has_queenside_castling_rights(chess.BLACK))
    return torch.tensor(bitboard)
   
games = parse('dataset/smol.pgn')
g1 = games[0]
bitboard = encode(g1.board(),chess.WHITE)
print(bitboard.shape)
print(bitboard)