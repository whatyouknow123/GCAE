import os
import argparse
import datetime
import torch
import torchtext.data as data
from w2v import *

from cnn_gate_aspect_model import CNN_Gate_Aspect_Text
from cnn_gate_aspect_model_atsa import CNN_Gate_Aspect_Text as CNN_Gate_Aspect_Text_atsa
import mydatasets as mydatasets
from getsemeval import get_semeval, get_semeval_target, read_yelp
import cnn_train
import json
from collections import defaultdict

good_lap_attributes = ['battery#operation_performance', 'battery#quality', 'company#general', 'cpu#operation_performance', 'display#design_features', 'display#general', 'display#operation_performance', 'display#quality', 'display#usability', 'graphics#general', 'graphics#quality', 'hard_disc#design_features', 'hard_disc#quality', 'keyboard#design_features', 'keyboard#general', 'keyboard#operation_performance', 'keyboard#quality', 'keyboard#usability', 'laptop#connectivity', 'laptop#design_features', 'laptop#general', 'laptop#miscellaneous', 'laptop#operation_performance', 'laptop#portability', 'laptop#price', 'laptop#quality', 'laptop#usability', 'memory#design_features', 'motherboard#quality', 'mouse#design_features', 'mouse#general', 'mouse#operation_performance', 'mouse#quality', 'mouse#usability', 'multimedia_devices#general', 'multimedia_devices#operation_performance', 'multimedia_devices#quality', 'optical_drives#quality', 'os#general', 'os#operation_performance', 'os#usability', 'power_supply#quality', 'shipping#quality', 'software#design_features', 'software#general', 'software#miscellaneous', 'software#operation_performance', 'software#usability', 'support#price', 'support#quality']

def parameter_parse():
    parser = argparse.ArgumentParser(description='CNN text classificer')

    # learning
    parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate [default: 0.001]')
    parser.add_argument('-l2', type=float, default=0, help='initial learning rate [default: 0]')
    parser.add_argument('-momentum', type=float, default=0.99, help='SGD momentum [default: 0.99]')
    parser.add_argument('-epochs', type=int, default=30, help='number of epochs for train [default: 30]')
    parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 32]')
    parser.add_argument('-grad_clip', type=float, default=5, help='max value of gradients')
    parser.add_argument('-lr_decay', type=float, default=0, help='learning rate decay')

    # logging
    parser.add_argument('-log-interval',  type=int, default=10,   help='how many steps to wait before logging training status [default: 10]')
    parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
    parser.add_argument('-save-interval', type=int, default=10000, help='how many steps to wait before saving [default:10000]')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')

    # data
    parser.add_argument('-shuffle', action='store_true', default=True, help='shuffle the data every epoch [default: True]')
    parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 300]')
    parser.add_argument('-aspect_embed_dim', type=int, default=300, help='number of aspect embedding dimension [default: 300]')
    parser.add_argument('-unif', type=float, help='Initializer bounds for embeddings', default=0.25)
    parser.add_argument('-embed_file', default='w2v', help='w2v or glove')
    parser.add_argument('-aspect_file', type=str, default='', help='aspect embedding')
    parser.add_argument('-years', type=str, default='14_15_16', help='data sets specified by the year, use _ to concatenate')
    parser.add_argument('-aspects', type=str, default='all', help='selected aspects, use _ to concatenate')
    parser.add_argument('-atsa', action='store_true', default=False)
    parser.add_argument('-r_l', type=str, default='r', help='restaurants or laptops')
    parser.add_argument('-use_attribute', action='store_true', default=False)
    parser.add_argument('-aspect_phrase', action='store_true', default=False)

    # model CNNs
    parser.add_argument('-model', type=str, default='CNN', help='Model name [default: CNN]')
    parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
    parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel [default: 100]')
    parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
    parser.add_argument('-att_dsz', type=int, default=100, help='Attention dimension [default: 100]')
    parser.add_argument('-att_method', type=str, default='concat', help='Attention method [default: concat]')

    ## CNN_CNN
    parser.add_argument('-lambda_sm', type=float, default=1.0, help='Lambda weight for sentiment loss [default: 1.0]')
    parser.add_argument('-lambda_as', type=float, default=1.0, help='Lambda weight for aspect loss [default: 1.0]')

    ## LSTM
    parser.add_argument('-lstm_dsz', type=int, default=300, help='LSTM hidden layer dimension size [default: 300]')
    parser.add_argument('-lstm_bidirectional', type=bool, default=True, help='is LSTM bidirecional [default: True]')
    parser.add_argument('-lstm_nlayers', type=int, default=1, help='the number of layers of LSTM [default: 1]')

    # device
    parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')

    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('-sentence', type=str, default=None, help='predict the sentence given')
    parser.add_argument('-target', type=str, default=None, help='predict the target given')

    parser.add_argument('-test', action='store_true', default=False, help='train or test')
    parser.add_argument('-verbose', type=int, default=0)
    parser.add_argument('-trials', type=int, default=1, help='the number of trials')
    parser.add_argument('-acsa_model_path', type=str, default="../model/acsa.pkl", help='ACSA model save path')
    parser.add_argument('-atsa_model_path', type=str, default="../model/atsa.pkl", help='ATSA model save path')
    parser.add_argument('-predict_result', type=str, default='../model/predict_result', help="predict result file")
    parser.add_argument('-text_vocab', type=str, default='../model/text_vocab', help="predict result file")
    parser.add_argument('-sentiment_vocab', type=str, default='../model/sentiment_vocab', help="predict result file")
    parser.add_argument('-aspect_vocab', type=str, default='../model/aspect_vocab', help="predict result file")

    args = parser.parse_args()

    # addition
    args.cuda = (not args.no_cuda) and torch.cuda.is_available();del args.no_cuda
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    return args


def load_semeval_data(text_field, as_field, sm_field, years, aspects, args, **kargs):
    if not args.atsa:
        semeval_train, semeval_test = get_semeval(years, aspects, args.r_l, args.use_attribute)
    else:
        semeval_train, semeval_test = get_semeval_target(years, args.r_l)

    predict_test = [{"aspect": "food",
                     "sentiment": "positive",
                     "sentence": "good food in cute - though a bit dank - little hangout, but service terrible"},
                    {"aspect": "service",
                     "sentiment": "negative",
                     "sentence": "good food in cute - though a bit dank - little hangout, but service terrible"},
                    {"aspect": "service",
                     "sentiment": "negative",
                     "sentence": "good food in cute - though a bit dank - little hangout, but service terrible"}
                    ]
    predict_data = mydatasets.SemEval(text_field, as_field, sm_field, predict_test)

    train_data, dev_data, mixed_data = mydatasets.SemEval.splits_train_test(text_field, as_field, sm_field,
                                                     semeval_train, semeval_test)

    text_field.build_vocab(train_data, dev_data)
    as_field.build_vocab(train_data, dev_data)
    sm_field.build_vocab(train_data, dev_data)
    train_iter, test_iter, mixed_test_iter, predict_iter = data.Iterator.splits(
                                (train_data, dev_data, mixed_data, predict_data),
                                batch_sizes=(args.batch_size, len(dev_data), len(mixed_data), len(predict_data)),
                                **kargs)

    return train_iter, test_iter, mixed_test_iter, predict_iter

def get_embedding(args, text_vocab, as_vocab, sm_field):
    text_vocab_list = sorted(text_vocab, key=text_vocab.get)
    as_vocab_list = sorted(as_vocab, key=as_vocab.get)
    if args.embed_file == 'w2v':
        print("Loading W2V pre-trained embedding...")
        word_vecs = load_w2v_embedding(text_vocab_list, args.unif, 300)
    elif args.embed_file == 'glove':
        print("Loading GloVe pre-trained embedding...")
        word_vecs = load_glove_embedding(text_vocab_list, args.unif, 300)
    else:
        raise(ValueError('Error embedding file'))
    print('# word initialized {}'.format(len(word_vecs)))

    print("Loading pre-trained aspect embedding...")
    if args.aspect_file == '':
        args.aspect_embedding = load_aspect_embedding_from_w2v(as_vocab_list, as_vocab, word_vecs)
        args.aspect_embed_dim = args.embed_dim
    else:
        args.aspect_embedding, args.aspect_embed_dim = load_aspect_embedding_from_file(as_vocab_list, args.aspect_file)

    args.embedding = torch.from_numpy(np.asarray(word_vecs, dtype=np.float32))
    args.aspect_embedding = torch.from_numpy(np.asarray(args.aspect_embedding, dtype=np.float32))

    print('# aspect embedding size: {}'.format(len(args.aspect_embedding)))
    return args


def save_vocab_dic(args, text_field, as_field, sm_field):
    flag = False
    try:
        with open(args.text_vocab, "w") as data:
            data.write(json.dumps(text_field.vocab.stoi, indent=4))
            print("save text vocab successfully")
        with open(args.aspect_vocab, "w") as data:
            data.write(json.dumps(as_field.vocab.stoi, indent=4))
            print("save aspect vocab successfully")
        with open(args.sentiment_vocab, "w") as data:
            data.write(json.dumps(sm_field.vocab.stoi, indent=4))
            print("save sentiment vocab successfully")
        flag = True
    except:
        flag = False
    return flag


def read_vocab_dic(args):
    flag = False
    text_vocab = {}
    aspect_vocab = {}
    sentiment_vocab = {}
    try:
        with open(args.text_vocab, "r") as data:
            text_vocab = json.load(data)
        with open(args.aspect_vocab, "r") as data:
            aspect_vocab = json.load(data)
        with open(args.sentiment_vocab, "r") as data:
            sentiment_vocab = json.load(data)
        flag = True
    except:
        flag = False
    return flag, text_vocab, aspect_vocab, sentiment_vocab

def load_data(args):
    # load data
    print("Loading data...")
    text_field = data.Field(lower=True, tokenize='moses')

    if not args.aspect_phrase:
        as_field = data.Field(sequential=False)
    else:
        print('phrase')
        as_field = data.Field(lower=True, tokenize='moses')

    sm_field = data.Field(sequential=False)
    years = [int(i) for i in args.years.split('_')]
    aspects = None
    if args.r_l == 'lap' and args.use_attribute:
        aspects = good_lap_attributes

    train_iter, test_iter, mixed_test_iter, predict_iter = load_semeval_data(text_field, as_field, sm_field, years,
                                                                             aspects, args,
                                                                             device=-1, repeat=False)

    print('# aspects: {}'.format(len(as_field.vocab.stoi)))
    print('# sentiments: {}'.format(len(sm_field.vocab.stoi)))
    args.embed_num = len(text_field.vocab)
    args.class_num = len(sm_field.vocab) - 1
    args.aspect_num = len(as_field.vocab)
    args = get_embedding(args, text_field.vocab.stoi, as_field.vocab.stoi, sm_field)
    save_flag = save_vocab_dic(args, text_field, as_field, sm_field)
    if save_flag:
        print("save text, aspect, sentiment vocab successful")
    else:
        print("fail to save text, aspect, sentiment vocab")
    return train_iter, test_iter, mixed_test_iter, predict_iter, args, text_field, as_field, sm_field



def choose_model(args, train_flag=True):
    model_path = "../model/absa.model"
    if args.model == 'CNN_Gate_Aspect' and not args.atsa:
        # GCAE
        model = CNN_Gate_Aspect_Text(args)
        model_path = args.acsa_model_path
    elif args.model == 'CNN_Gate_Aspect' and args.atsa:
        # CNN on target expressions
        model = CNN_Gate_Aspect_Text_atsa(args)
        model_path = args.atsa_model_path
    else:
        raise(ValueError('Error Model'))

    if not train_flag:
        print('\nLoading model from {}...'.format(model_path))
        model.load_state_dict(torch.load(model_path))

    model = model.cuda()
    return model, model_path


def train_model(args):
    train_iter, test_iter, mixed_test_iter, predict_iter, args, text_field, as_field, sm_field = load_data(args)
    n_trials = args.trials
    accuracy_trials = []
    time_stamps_trials = []
    train = cnn_train
    model, model_path = choose_model(args)
    for t in range(n_trials):
        # train or predict
        if args.test:
            try:
                train.eval(test_iter, model, args)
            except Exception as e:
                print("\nSorry. The test dataset doesn't  exist.\n")
        else:
            print()
            acc, time_stamps = train.train(train_iter, test_iter, mixed_test_iter, model, args, text_field, as_field, sm_field, predict_iter)
            accuracy_trials.append([acc[0], acc[1]])   # accuracy on test, accuracy on mixed
            time_stamps_trials.append(time_stamps)
            torch.save(model.state_dict(), model_path)
            print(model)
            save_accuracy(accuracy_trials, time_stamps_trials)

    return accuracy_trials, time_stamps_trials


def predict(args, sentences, targets, text_vocab, as_vocab, sm_vocab):
    args.embed_num = len(text_vocab)
    args.class_num = len(sm_vocab) - 1
    args.aspect_num = len(as_vocab)
    args = get_embedding(args, text_vocab, as_vocab, sm_vocab)
    model, model_path = choose_model(args, False)
    train = cnn_train
    sm_field = sorted(sm_vocab, key=sm_vocab.get)
    labels = train.predict(model, sentences, text_vocab, targets, as_vocab, sm_field)
    for index, sentence in enumerate(sentences):
        target = targets[index]
        label = labels[index]
        print('\n[Text]  {} [Target] {} [Label] {}\n'.format(sentence, target, label))
    return labels


def save_accuracy(accuracy_trials, time_stamps_trials):
    print(accuracy_trials)
    accuracy_trials = np.array(accuracy_trials)
    means = accuracy_trials.mean(0)
    print('{:.2f}'.format(means[0]))
    print('{:.2f}'.format(means[1]))

    with open('time_stamps', 'w') as fopen:
        for trials in time_stamps_trials:
            for acc, _ in trials:
                fopen.write('{:.4f} '.format(acc))
            fopen.write('\n')
            for _, dtime in trials:
                fopen.write('{:.4f} '.format(dtime))
            fopen.write('\n')

if __name__ == "__main__":
    pameters = parameter_parse()
    #train_model(pameters)

    flag, text_vocab, aspect_vocab, sentiment_vocab = read_vocab_dic(pameters)
    if flag:
        print("read text vocab: %d, aspect vocab: %d, sentiment vocab: %d" % (len(text_vocab), len(aspect_vocab), len(sentiment_vocab)))
        sentences = ["the food is bad, the service is excellent. the price is too expensive, it is not deserved!", "the food is healthful, the service is excellent. the price is too expensive, it is not deserved!", "the food is healthful, the service is excellent. the price is too expensive, it is not deserved!"]
        targets = ["food", "service", 'price']
        labels = predict(pameters, sentences, targets, text_vocab, aspect_vocab, sentiment_vocab)
        print(labels)


