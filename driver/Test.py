import sys
sys.path.extend(["../../", "../", "./"])
import random
import time
import pickle
import argparse
from data.Config import *
from modules.DialogDP import *
from modules.Decoder import *
from script.evaluation import *
from modules.GlobalEncoder import *
from data.BertTokenHelper import *
from modules.BertModelTune import *
from modules.SPEncoder import SPEncoder
from driver.Train import predict

from torch.cuda.amp import autocast as autocast
from torch.cuda.amp.grad_scaler import GradScaler

if __name__ == '__main__':
    print("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))
    ### seed
    random.seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.manual_seed(666)

    # torch version
    print("Torch Version: ", torch.__version__)

    # gpu state
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: ", torch.backends.cudnn.enabled)

    # args
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='examples/default.cfg')
    argparser.add_argument('--model', default='BaseParser')
    argparser.add_argument('--thread', default=1, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    test_instances = read_corpus(config.test_file)

    vocab = pickle.load(open(config.load_vocab_path, 'rb'))
    dp_model = torch.load(config.load_model_path)

    torch.set_num_threads(args.thread)

    tok_helper = BertTokenHelper(config.load_bert_path)
    bert_extractor = BertExtractor(config.load_bert_path, config, tok_helper)

    global_encoder = GlobalEncoder(vocab, config, bert_extractor)
    state_encoder = StateEncoder(vocab, config)
    sp_encoder = SPEncoder(vocab, config)
    decoder = Decoder(vocab, config)

    global_encoder.mlp_words.load_state_dict(dp_model["mlp_words"])
    global_encoder.rescale.load_state_dict(dp_model["rescale"])
    global_encoder.edu_GRU.load_state_dict(dp_model["edu_GRU"])
    sp_encoder.load_state_dict(dp_model["sp_encoder"])
    state_encoder.load_state_dict(dp_model["state_encoder"])
    decoder.load_state_dict(dp_model["decoder"])

    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    # print(global_encoder)
    print(state_encoder)
    print(decoder)

    if config.use_cuda:
        torch.backends.cudnn.enabled = True

        global_encoder.cuda()
        sp_encoder.cuda()
        state_encoder.cuda()
        decoder.cuda()

    parser = DialogDP(global_encoder, state_encoder, sp_encoder, decoder, config)
    with torch.no_grad():
        predict(test_instances, parser, vocab, config, tok_helper, config.test_file + ".out")
    print("Test:")
    evaluation(config.test_file, config.test_file + ".out")
