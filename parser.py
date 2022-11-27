import chess.pgn

def parse(path):
    pgn = open(path)
    games=[]
    c = 0
    while(True):
        g = chess.pgn.read_game(pgn)
        if g is None: break
        c+=1
        if g.headers["Result"] != "1/2-1/2" : games.append(g) #ignore games that end in a draw 
        print('games parsed %d'%(c))

    print('done parsing %d games'%(c))

    print(f"games loaded = {len(games)}")
    pgn.close()
    return(games)
