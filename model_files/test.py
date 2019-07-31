import os
import argparse
import datetime
import torch
import torchtext.data as data
import torch.nn.functional as F
from w2v import *
import json

from cnn_gate_aspect_model import CNN_Gate_Aspect_Text
from cnn_gate_aspect_model_atsa import CNN_Gate_Aspect_Text as CNN_Gate_Aspect_Text_atsa



import mydatasets as mydatasets
from getsemeval import get_semeval, get_semeval_target, read_yelp
import cnn_train


def load_semeval_data(text_field, as_field, sm_field, years, aspects, flag=False, **kargs):
    if not flag:
        semeval_train, semeval_test = get_semeval(years, aspects)
    else:
        semeval_train, semeval_test = get_semeval_target(years)

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

    print(train_data)
    print(dev_data)
    text_field.build_vocab(train_data, dev_data)
    as_field.build_vocab(train_data, dev_data)
    sm_field.build_vocab(train_data, dev_data)
    train_iter, test_iter, mixed_test_iter, predict_iter = data.Iterator.splits(
                                (train_data, dev_data, mixed_data, predict_data),
                                batch_sizes=(32, len(dev_data), len(mixed_data), len(predict_data)),
                                **kargs)
    return train_iter, test_iter, mixed_test_iter, predict_iter


def eval(data_iter, model, text_filed=None, aspect_field=None, sm_field=None):
    model.eval()
    corrects, avg_loss = 0, 0
    loss = None
    write_result = open("../model_predict_result", "a")
    for batch in data_iter:
        feature, aspect, target = batch.text, batch.aspect, batch.sentiment
        feature = feature.data.t()
        if not False:
            aspect = aspect.data.unsqueeze(0)
        aspect = aspect.data.t()
        target = target.data.sub(1)  # batch first, index align
        if torch.cuda.is_available():
            feature, aspect, target = feature.cuda(), aspect.cuda(), target.cuda()
        logit, _, _ = model(feature, aspect)
        logit_max = torch.max(logit, 1)[1].view(target.size()).data
        if text_filed and aspect_field and sm_field:
            if feature.size()[0] == aspect.size()[0] and aspect.size()[0] == target.size()[0] and target.size()[0] == logit_max.size()[0]:
                origin_target = [sm_field.vocab.itos[int(target[i]) + 1] for i in range(target.size()[0])]
                origin_logit = [sm_field.vocab.itos[int(logit_max[i]) + 1] for i in range(logit_max.size()[0])]
                for index in range(feature.size()[0]):
                    feature_one = feature[index]
                    aspect_one = aspect[index]
                    origin_feature = " ".join([text_filed.vocab.itos[int(feature_one[i])] for i in range(feature_one.size()[0])])
                    origin_feature = origin_feature.strip(" <pad>")
                    origin_aspect = [aspect_field.vocab.itos[int(aspect_one[i])] for i in range(aspect_one.size()[0])]
                    print("origin feature: ", origin_feature)
                    print("origin aspect: ", origin_aspect)
                    print("origin target: ", origin_target[index])
                    print("origin logit: ", origin_logit[index])
                    write_str = origin_feature + "\t" + origin_aspect[0] + "\t" + origin_target[index] + "\t" + origin_logit[index] + "\n"
                    write_result.write(write_str)
        loss = F.cross_entropy(logit, target, size_average=False)
        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = loss.item()/size
    accuracy = 100.0 * corrects/size
    model.train()
    #if args.verbose > 1:
    #    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(
    #       avg_loss, accuracy, corrects, size))
    return accuracy


n_trials = 1
accuracy_trials = []
time_stamps_trials = []

# load data
print("Loading data...")
text_field = data.Field(lower=True, tokenize='moses')

if not False:
    as_field = data.Field(sequential=False)
else:
    print('phrase')
    as_field = data.Field(lower=True, tokenize='moses')

sm_field = data.Field(sequential=False)
years = [int(i) for i in "14".split('_')]
aspects = None
#if args.r_l == 'lap' and args.use_attribute:
#    aspects = good_lap_attributes

train_iter, test_iter, mixed_test_iter, predict_iter = load_semeval_data(text_field, as_field, sm_field, years, aspects,
                                                          device=-1, repeat=False)

print('# aspects: {}'.format(len(as_field.vocab.stoi)))
print('# sentiments: {}'.format(len(sm_field.vocab.stoi)))
print('# aspect is: ', as_field.vocab.freqs)
print('# sem is: ', sm_field.vocab.freqs)
print("# aspect itos: ", as_field.vocab.itos)
print("# aspect stoi: ", as_field.vocab.stoi)

#print(sm_field.vocab.vectors)
print("# sem itos: ", sm_field.vocab.itos)
print("# sem stoi: ", sm_field.vocab.stoi)

print("# test itos: ", text_field.vocab.itos)
print("# test stoi: ", text_field.vocab.stoi)
sm_field = sm_field.vocab.stoi
sm_str = json.dumps(sm_field)
print(sm_str)
sm_filed = json.loads(sm_str)
values = sorted(sm_filed, key=sm_filed.get)
print(values)
'''
word_list = ["the", "and", "in"]
word_vec = load_glove_embedding(word_list, 0.25, 300)
for index, word in enumerate(word_list):
    print(word)
    print(word_vec[index])

model_path = "../model/atsa.model"
model = torch.load(model_path, map_location='cpu')
#model = model.cuda()
print(test_iter)
eval(test_iter, model)


test = torch.Tensor([[1,2,3,4],[1,2,3,4],[1,2,3,4]])
print(test)
values = ["1", "2", "3", "4", "5", "6", "7"]
for each in test:
    print(each)
    print(each.size())
    sentence = " ".join([values[int(each[index])] for index in range(each.size()[0])])
    print(sentence)

value = "the best chinese food uptown ! <pad> <pad>"
print(value.strip(" <pad>"))
'''









