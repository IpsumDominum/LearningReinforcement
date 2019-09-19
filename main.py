from GymAtari_wrapper import *
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="input arguments")
    parser.add_argument('-env_name',type=str,default="Breakout-v0",help="name of the env")
    parser.add_argument('-state_size',type=int,default=84,help="size of resized frame")

    parser.add_argument('-frame_size',type=int,default=4,help="size of resized frame")
    parser.add_argument('-agent',type=str,default="DQN",help="agent specification")
    parser.add_argument('-render',action='store_true',default=True,help="render or not")
    parser.add_argument('-train',action='store_true',default=True,help="test or not")
    parser.add_argument('-load_path',type=str,default="checkpoint.pth",help='load path')
    args = parser.parse_args()
    Game = GymAtari(args.env_name,args.state_size,args.frame_size,args.agent,args.render,args.train,args.load_path)
    Game.main_loop(2000)
    
