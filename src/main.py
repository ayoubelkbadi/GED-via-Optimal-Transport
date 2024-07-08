from utils import tab_printer
from trainer import Trainer
from param_parser import parameter_parser
import random
import os
import numpy as np
import torch
import sys
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def main():
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a SimGNN model.
    """
    seed_torch(1)
    args = parameter_parser()
    tab_printer(args)
    trainer = Trainer(args)
    if args.GW and args.model_name == "GEDGW":
        trainer.process('test')
        trainer.process('test2')
        trainer.path_score_my('test',test_k=1)
    else:
        if args.model_epoch_start > 0:
            trainer.load(args.model_epoch_start)
        if args.model_train == 1:
            for epoch in range(args.model_epoch_start, args.model_epoch_end):
                trainer.cur_epoch = epoch
                trainer.fit()
                trainer.save(epoch + 1)
                trainer.score('test')
        else:
            trainer.cur_epoch = args.model_epoch_start
            if args.model_name == "NOAH":
                trainer.score_my('test',test_k=0)
                trainer.score_my('test2',test_k=0)
            elif not args.greedy:
                trainer.score_my('test')
                trainer.score_my('test2')
            if args.path:
                if args.model_name in ["GPN","SimGNN","TaGSim"]:
                    print("Warning: No path can be generated")
                    sys.exit()
            if args.greedy:
                trainer.path_score_my('test', test_k=1)
            elif args.model_name=="NOAH":
                trainer.path_score_my('test', test_k=0)
            else:
                #GedGNN and Ours
                trainer.path_score_my('test',test_k=100)                


if __name__ == "__main__":
    main()
