from utils import *
from stockfish import Stockfish
import pickle
import argparse
from tqdm import tqdm

TEST_BOARD = [[-479, -280, -320, -929, -60000], [-100, -100, -100, -100, -100], [0, 0, 0, 0, 0], [100, 100, 100, 100, 100], [479, 280, 320, 929, 60000]]

parser = argparse.ArgumentParser()
parser.add_argument('--total', default=10)
parser.add_argument('--i',default=0)
args = parser.parse_args()
if __name__ == "__main__":

    # stockfish = Stockfish("/usr/local/Cellar/stockfish/14/bin/stockfish")
    # stockfish.set_depth(10)

    total = int(args.total )
    i = int(args.i )

    turns = pickle.load(open("/Users/shiningsunnyday/Desktop/2021-2022/Fall Quarter/AA 228/Final Project/meta-minichess/temp/checkpoint_0.pth.tar.examples", "rb"))[0]
    chunk_size = (len(turns) + total - 1)//total
    print(chunk_size)
    turns = turns[i * chunk_size : (i+1) * chunk_size]

    print(len(turns), "turns in total")


    boards = np.empty((1, 5, 5))
    
    y = np.empty((1,))
    pis = np.empty((942,))

    turns = preprocess(turns)
    

    for (_, (b, pi, w)) in tqdm(enumerate(turns)):
        turn = 'w' if w > 0 else 'b'

        # print("turn:",w)
        # b_ = convert_mini(b)
        # print(b_)
        # board = bit_mask(b_).flatten()

        # B=bit_mask(b_)


        # assert board.size == (12 * 5 * 5)
        # print("got bit mask")


        # score = evaluate_mini(b, stockfish, turn)

        score = np.sum(b) # this is for white, later mult by -1 if training black critic



        boards = np.vstack((boards, np.array(b).reshape(-1, 5, 5)))  
            
        y = np.vstack((y, score))
        pis = np.vstack((pis, pi))
        




  

    train_boards, test_boards, train_pis, test_pis, train_y, test_y = postprocess(boards, pis, y)

    np.save("data/total_%d_i_%d_train_x.npy" % (total, i), train_boards)    
    np.save("data/total_%d_i_%d_test_x.npy" % (total, i), test_boards)
    np.save("data/total_%d_i_%d_train_pis.npy" % (total, i), train_pis)    
    np.save("data/total_%d_i_%d_test_pis.npy" % (total, i), test_pis)
    np.save("data/total_%d_i_%d_train_y.npy" % (total, i), train_y)
    np.save("data/total_%d_i_%d_test_y.npy" % (total, i), test_y)
    


