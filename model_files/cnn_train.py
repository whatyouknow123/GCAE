import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
import mydatasets
import torchtext.data as data

def train(train_iter, dev_iter, mixed_test_iter, model, args, text_field, aspect_field, sm_field, predict_iter):
    time_stamps = []

    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.l2, lr_decay=args.lr_decay)

    steps = 0
    model.train()
    start_time = time.time()
    dev_acc, mixed_acc = 0, 0
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            feature, aspect, target = batch.text, batch.aspect, batch.sentiment

            feature = feature.data.t()
            if len(feature) < 2:
                continue
            if not args.aspect_phrase:
                aspect = aspect.data.unsqueeze(0)
            aspect = aspect.data.t()
            target.data.sub_(1)  # batch first, index align

            if args.cuda:
                feature, aspect, target = feature.cuda(), aspect.cuda(), target.cuda()

            optimizer.zero_grad()
            logit, _, _ = model(feature, aspect)

            loss = F.cross_entropy(logit, target)
            loss.backward()

            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                if args.verbose == 1:
                    sys.stdout.write(
                        '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                                 loss.item(),
                                                                                 accuracy,
                                                                                 corrects,
                                                                                 batch.batch_size))


            if steps % args.save_interval == 0:
                if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
                save_prefix = os.path.join(args.save_dir, 'snapshot')
                save_path = '{}_steps{}.pt'.format(save_prefix, steps)
                torch.save(model, save_path)

        if epoch == args.epochs:
            dev_acc, _, _ = eval(dev_iter, model, args, text_field, aspect_field,sm_field)
            if mixed_test_iter:
                mixed_acc, _, _ = eval(mixed_test_iter, model, args)
            else:
                mixed_acc = 0.0

            if args.verbose == 1:
                delta_time = time.time() - start_time
                print('\n{:.4f} - {:.4f} - {:.4f}'.format(dev_acc, mixed_acc, delta_time))
                time_stamps.append((dev_acc, delta_time))
                print()
    return (dev_acc, mixed_acc), time_stamps


def eval(data_iter, model, args, text_filed=None, aspect_field=None, sm_field=None):
    model.eval()
    corrects, avg_loss = 0, 0
    loss = None
    write_result = open(args.predict_result, "a")
    for batch in data_iter:
        feature, aspect, target = batch.text, batch.aspect, batch.sentiment
        feature = feature.data.t()
        if not args.aspect_phrase:
            aspect = aspect.data.unsqueeze(0)
        aspect = aspect.data.t()
        target = target.data.sub(1)  # batch first, index align
        if args.cuda:
            feature, aspect, target = feature.cuda(), aspect.cuda(), target.cuda()
        logit, pooling_input, relu_weights = model(feature, aspect)
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
                    #print("origin feature: ", origin_feature)
                    #print("origin aspect: ", origin_aspect)
                    #print("origin target: ", origin_target[index])
                    #print("origin logit: ", origin_logit[index])
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
    if args.verbose > 1:
        print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(
           avg_loss, accuracy, corrects, size))
    return accuracy, pooling_input, relu_weights


def predict(model, sentences, text_field, targets, as_field, sm_field, max_offset_len=100):
    model.eval()
    aspects_index = []
    for target in targets:
        aspects_index.append(as_field[target])
    aspect = torch.LongTensor(aspects_index)
    sentences_index = []
    for sentence in sentences:
        sentence_index = [text_field.setdefault(token, 1) for token in sentence.strip().split()]
        if len(sentence_index) < max_offset_len:
            sentence_index.extend([1 for i in range(max_offset_len - len(sentence_index))])
        sentences_index.append(sentence_index)
    predict_iter = torch.LongTensor(sentences_index)
    print("aspect is: ", aspect)
    print("predict iter: ", predict_iter)
    #predict_iter = predict_iter.data.t()
    aspect = aspect.data.unsqueeze(0)
    aspect = aspect.data.t()
    aspect = aspect.cuda()
    predict_iter = predict_iter.cuda()
    logit, _, _ = model(predict_iter, aspect)
    print(logit)
    label_index = torch.max(logit, 1)[1].data
    print(label_index)
    if label_index.size()[0] != len(targets):
        print("predict result mismatch given data")
    labels = [sm_field[int(label_index[i]) + 1] for i in range(label_index.size()[0])]
    print(labels)
    return labels

    


