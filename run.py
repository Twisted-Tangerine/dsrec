# here put the import lib
import os
import argparse
import torch

from src.dataset import Generator, Seq2SeqGenerator, BertGenerator
from src.trainers.sequence_trainer import SeqTrainer
from src.utils.helpers import set_seed
from src.utils.logger import Logger
from src.config import build_parser


parser = build_parser()

torch.autograd.set_detect_anomaly(True)

args = parser.parse_args()
set_seed(args.seed) # fix the random seed
args.output_dir = os.path.join(args.output_dir, args.dataset)
args.pretrain_dir = os.path.join(args.output_dir, args.pretrain_dir)
args.output_dir = os.path.join(args.output_dir, args.model_name)
args.output_dir = os.path.join(args.output_dir, args.check_path)    # if check_path is none, then without check_path
args.sem_emb_path = os.path.join("data/"+args.dataset+"/", "{}.pkl".format(args.sem_emb))


def main():

    log_manager = Logger(args)  # initialize the log manager
    logger, writer = log_manager.get_logger()    # get the logger
    args.now_str = log_manager.get_now_str()

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")


    os.makedirs(args.output_dir, exist_ok=True)

    # generator is used to manage dataset
    if args.model_name in ['gru4rec', "dsrec_gru4rec"]:
        generator = Generator(args, logger, device)
    elif args.model_name in ['bert4rec', "dsrec_bert4rec"]:
        generator = BertGenerator(args, logger, device)
    elif args.model_name in ['sasrec', "dsrec_sasrec"]:
        generator = Seq2SeqGenerator(args, logger, device)
    else:
        raise ValueError

    trainer = SeqTrainer(args, logger, writer, device, generator)

    if args.do_test:
        trainer.test()
    elif args.do_emb:
        trainer.save_item_emb()
        trainer.save_user_emb()
    elif args.do_group:
        trainer.test_group()
    else:
        trainer.train()

    log_manager.end_log()   # delete the logger threads



if __name__ == "__main__":

    main()
