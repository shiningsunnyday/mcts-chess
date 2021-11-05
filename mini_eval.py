from utils import *
from stockfish import Stockfish
import pickle

TEST_BOARD = [[-479, -280, -320, -929, -60000], [-100, -100, -100, -100, -100], [0, 0, 0, 0, 0], [100, 100, 100, 100, 100], [479, 280, 320, 929, 60000]]

if __name__ == "__main__":
    stockfish = Stockfish("/usr/local/Cellar/stockfish/14/bin/stockfish")
    stockfish.set_depth(2)

    turns = pickle.load(open("data/checkpoint_0.pth.tar.examples", "rb"))[0]
    boards = np.empty((1, 5, 5))
    y = np.empty((1,))

    turns = preprocess(turns)
    

    for (i, (b, _, w)) in enumerate(turns):
        turn = 'w' if w > 0 else 'b'

        # print("turn:",w)
        # b_ = convert_mini(b)
        # print(b_)
        # board = bit_mask(b_).flatten()

        # B=bit_mask(b_)


        # assert board.size == (12 * 5 * 5)
        # print("got bit mask")


        score = evaluate_mini(b, stockfish, turn)


        boards = np.vstack((boards, np.array(b).reshape(-1, 5, 5)))  
            
        y = np.vstack((y, score))
        print(boards.shape, y.shape)  


  

    train_boards, test_boards, train_y, test_y = postprocess(boards, y)

    np.save("data/checkpoint_0_train_x.npy", train_boards)    
    np.save("data/checkpoint_0_test_x.npy", test_boards)
    np.save("data/checkpoint_0_train_y.npy", train_y)
    np.save("data/checkpoint_0_test_y.npy", test_y)
    


