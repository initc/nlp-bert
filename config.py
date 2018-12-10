import logging
import logging.config
import argparse

def get_logging(filename):

    LOG_CONFIG = {
        'version': 1,
        'formatters': {
            'default': {'format': '%(asctime)s - %(levelname)s - %(message)s', 'datefmt': '%Y-%m-%d %H:%M:%S'}
        },
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'default',
                # 'stream': 'ext://sys.stdout'
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'default',
                'filename': filename,
                # 'maxBytes': 1024,
                # 'backupCount': 3
            }
        },
        'loggers': {
            'default': {
                'level': 'DEBUG',
                'handlers': ['file', 'console']
            }
        },
        'disable_existing_loggers': True
    }
    logging.config.dictConfig(LOG_CONFIG)
    return logging.getLogger('default')

def generate_args():
    parser = argparse.ArgumentParser()

    ## Other parameters
    parser.add_argument("--data", default="data", type=str, help="MSmarco dev data")
    parser.add_argument("--path", default="data", type=str, help="path(s) to model file(s), colon separated")
    parser.add_argument("--valid-batch-size", default=2, type=int, help="Total batch size for predictions.")
    parser.add_argument("--max-passage-tokens", default=200, type=int,
                        help="The maximum total input passage length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max-query-tokens", default=50, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument('--do-lower-case',
                        default=False, action='store_true',
                        help='whether case sensitive')
    parser.add_argument('--question-first',
                        default=False, action='store_true',
                        help='whether case sensitive')
    parser.add_argument('--log-name', type=str, default="logfile.log")
    parser.add_argument('--threshold', type=int, default=0.36)
    parser.add_argument('--loss-interval', type=int, default=500, metavar='N',
                       help='validate every N updates')
    parser.add_argument("--pre-dir", type=str,
                        help="where the pretrained checkpoint")

    args = parser.parse_args()
    return args

def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument("--save-dir", default="checkpoints", type=str,
                        help="path to save checkpoints")

    ## Other parameters
    parser.add_argument("--data", default="data", type=str, help="MSmarco train and dev data")
    parser.add_argument("--origin-data", default="data", type=str, help="MSmarco train and dev data, will be tokenizer")
    parser.add_argument("--path", default="data", type=str, help="path(s) to model file(s), colon separated")
    parser.add_argument("--save", default="checkpoints/MSmarco", type=str, help="path(s) to model file(s), colon separated")
    parser.add_argument("--pre-dir", type=str,
                        help="where the pretrained checkpoint")
    parser.add_argument("--log-name", type=str,
                        help="where logfile")
    parser.add_argument("--max-passage-tokens", default=200, type=int,
                        help="The maximum total input passage length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max-query-tokens", default=50, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument('--gradient-accumulation-steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--train-batch-size", default=2, type=int, help="Total batch size for training.")
    parser.add_argument("--valid-batch-size", default=2, type=int, help="Total batch size for predictions.")
    parser.add_argument("--lr", default=6.25e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num-train-epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup-proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--do-lower-case',
                        default=False, action='store_true',
                        help='whether case sensitive')
    parser.add_argument('--question-first',
                        default=False, action='store_true',
                        help='whether case sensitive')
    parser.add_argument('--threshold', type=int, default=0.36)
    parser.add_argument('--logfile', type=str, default="logfile.log")
    parser.add_argument('--validate-updates', type=int, default=2000, metavar='N',
                       help='validate every N updates')
    parser.add_argument('--loss-interval', type=int, default=500, metavar='N',
                       help='validate every N updates')
    parser.add_argument('--merge-batch', action='store_true', default=False,
                       help='meger mutil batch into one until 90% batch have done!')
    parser.add_argument('--merge-size', type=int, default=2,
                       help='meger two batch into one until 90% batch have done!')
    args = parser.parse_args()
    # logging = logging.getlogging(args.log_name)
    # logging = config.get_logging("log/msmarco_1204_small.log")
    # first make corpus
    # tokenizer = BertTokenizer.build_tokenizer(args)
    # make_msmarco(args, tokenizer)

    return args