import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import time


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(epoch, train_loader, model, model_t, optimizer, opt):
    model_t.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    for idx, data in enumerate(train_loader):
        input, target, index = data
        data_time.update(time.time() - end)
        input = input.float()
        input = input.cuda()
        target = target.cuda()
        index = index.cuda()
        feat_s, logit_s = model(input, is_feat=True, preact=False)
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=False)
            feat_t = [f.detach() for f in feat_t]
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)

        f_s = regress_s(feat_s[opt.hint_layer])
        f_t = feat_t[opt.hint_layer]

        loss=nn.MSELoss(f_s, f_t)
        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    return top1.avg, losses.avg


class ConvReg(nn.Module):
    """Convolutional regression for FitNet"""
    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1+s_H-t_H, 1+s_W-t_W))
        else:
            raise NotImplemented('student size {}, teacher size {}'.format(s_H, t_H))
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)



def language_model_select(model, device, primer='a photo of a {}'):
    if model not in ['clip', 'bert', 'roberta_l']:
        raise NotImplementedError(
            'Language model {} not available!'.format(model))
    if model == 'clip':
        return ClipLanguageModel(primer, device)
    if model == 'bert':
        return BertLanguageModel(primer)
    if model == 'roberta_l':
        return RobertaLargeLanguageModel(primer)


class ClipLanguageModel(torch.nn.Module):
    def __init__(self, primer, device):
        super(ClipLanguageModel, self).__init__()
        self.name = 'CLIP-Language'
        self.primer = primer
        self.tokenizer = clip.tokenize
        self.model, _ = clip.load("ViT-B/32", device=device, jit=False)
        self.out_dim = 512

    def forward(self, text, device, skip_primer=False):
        if skip_primer:
            primed_tokens = text
        else:
            primed_tokens = [self.primer.format(x) for x in text]
        primed_tokens = self.tokenizer(primed_tokens)
        language_embeds = self.model.encode_text(primed_tokens.to(device))
        return language_embeds.type(torch.float32)


class RobertaLargeLanguageModel(torch.nn.Module):
    def __init__(self, primer, **kwargs):
        super(RobertaLargeLanguageModel, self).__init__()
        from transformers import RobertaTokenizer, RobertaModel
        self.name = 'Roberta-Large'
        self.primer = primer
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.model = RobertaModel.from_pretrained('roberta-large')
        self.out_dim = 1024

    def forward(self, text, device, skip_primer=False):
        if skip_primer:
            primed_tokens = text
        else:
            primed_tokens = [self.primer.format(x) for x in text]
        primed_tokens = self.tokenizer(primed_tokens,
                                       return_tensors='pt',
                                       padding=True,
                                       truncation=True).to(device)
        language_embeds = self.model(**primed_tokens).pooler_output
        return language_embeds.type(torch.float32)


class BertLanguageModel(torch.nn.Module):
    def __init__(self, primer, **kwargs):
        super(BertLanguageModel, self).__init__()
        from transformers import BertTokenizer, BertModel
        self.name = 'BERT'
        self.primer = primer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.out_dim = 1024

    def forward(self, text, device, skip_primer=False):
        if skip_primer:
            primed_tokens = text
        else:
            primed_tokens = [self.primer.format(x) for x in text]
        primed_tokens = self.tokenizer(primed_tokens,
                                       return_tensors='pt',
                                       padding=True,
                                       truncation=True).to(device)
        language_embeds = self.model(**primed_tokens).pooler_output
        return language_embeds.type(torch.float32)