from model_train import *


def build_parser():
    parser = ArgumentParser(prog="ESPCN Dataset Preparation.")
    parser.add_argument("-t", "--model", required=False, type=str,default='inferance',help="Required. train / inferance .")
    return parser


def main(args):
    model_config = Model_Config()
    model_object = Model_Train(model_config)
    if args.model == "train":
        model_object.model_train()
    else:
        model_object.inferance()


if __name__ == '__main__':
    args = build_parser().parse_args()
    espcn_model = main(args)
