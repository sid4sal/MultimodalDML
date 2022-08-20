import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import time
import pretrainedmodels as ptm


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


# placeholder for batch features
features = {}    
##### HELPER FUNCTION FOR FEATURE EXTRACTION
def get_features(name):
    def hook(model, input, output):
        global features
        features[name] = output.detach()
    return hook   

def train(epoch, dataloaders, model, model_t, optimizer, opt):
    ##### REGISTER HOOK
    model.global_pool.register_forward_hook(get_features('feat_s'))

    model_t.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    for idx, data in enumerate(dataloaders['training']):
        input = data[1]['image']
        target = data[0]

        data_time.update(time.time() - end)
        input = input.float()
        input = input.cuda()
        target = target.cuda()

        logit_s = model(input, device=opt.device)
        feat_s = features['feat_s']

        feat_t = precompute_language_embeds(opt, model_t, dataloaders['evaluation'], 
                                            ptm.__dict__['resnet50'](pretrained='imagenet').to(opt.device))
        
        #with torch.no_grad():
        #    feat_t = model_t(input_t, device=opt.device)
        #    feat_t = [f.detach() for f in feat_t]
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


def precompute_language_embeds(opt, language_model,
                                   dataloader,
                                   pseudoclass_generator=None):
    if opt.language_pseudoclass:
        classlevel_relabels, sample_relabels = relabel(
            pseudoclass_generator,
            dataloader,
            opt.device,
            topk=opt.language_pseudoclass_topk)

        language_embeds = reembed_in_language(
            language_model, classlevel_relabels,
            sample_relabels, opt.device)

        language_embeds = language_embeds.to(opt.device)
        language_embeds = language_embeds.permute(1, 0, 2)
        print('Retrieved {} language embeddings!'.format(
            language_embeds.shape[0] * language_embeds.shape[1]))
    else:
        language_embeds = reembed_dict_in_language(
            language_model, dataloader.dataset.language_conversion,
            opt.device)
        print('Retrieved {} language embeddings!'.format(
            len(language_embeds)))
    return language_embeds


#############################################################################
def adjust_text(input_text, maxlen=30):
    text = ''
    count = 0
    for p, c in enumerate(input_text.split(' ')):
        if p:
            text += ' '
        if count > maxlen and len(text) > 0:
            text += '\n'
            count -= maxlen
        text += c
        count += len(c)
    return text


def reembed_dict_in_language(language_model, label_dict, device):
    print('Getting language embeddings...')

    sorted_values = list(label_dict.values())
    unique_labs = {key: None for key in np.unique(sorted_values)}

    reembed_collect = []
    with torch.no_grad():
        language_embeds = language_model(list(unique_labs.keys()), device,
                                         False).cpu()
        unique_labs = {
            key: language_embed
            for key, language_embed in zip(unique_labs.keys(), language_embeds)
        }

    return {key: unique_labs[value] for key, value in label_dict.items()}


def reembed_in_language(language_model, reassigns_topk, device):
    print('Getting language embeddings...')
    unique_labs = {key: None for key in np.unique(reassigns_topk)}
    reembed_collect = []
    _ = language_model.eval()
    with torch.no_grad():
        language_embeds = language_model(list(unique_labs.keys()), device,
                                         False).cpu()
        unique_labs = {
            key: language_embed
            for key, language_embed in zip(unique_labs.keys(), language_embeds)
        }

    def match(inp):
        return [unique_labs[i] for i in inp]

    reembed_collect = list(map(match, reassigns_topk))
    return torch.stack([torch.stack(x) for x in reembed_collect])


def relabel(model,
            dataloader,
            device,
            datapath='',
            full_label=False,
            topk=5,
            overlap=True):
    was_training = model.training
    _ = model.eval()

    crop_size = dataloader.dataset.crop_size
    base_size = dataloader.dataset.base_size
    dataloader.dataset.crop_size = [299, 299]
    dataloader.dataset.base_size = 320
    dataloader.dataset.provide_transforms()

    if overlap:
        assert topk > 1, 'If you want label overlap, please set topk > 1!'

    with open(datapath + 'imagenet_synsets.txt', 'r') as f:
        imagenet_synsets = f.readlines()
    imagenet_classes = [x.strip() for x in imagenet_synsets]
    imagenet_splits = [line.split(' ') for line in imagenet_synsets]
    key_to_classname = {
        spl[0]: ' '.join(spl[1:]).replace('\n', '')
        for spl in imagenet_splits
    }

    with open(datapath + 'imagenet_classes.txt', 'r') as f:
        imagenet_classes = f.readlines()
    abstract_imagenet_classes = [
        x.strip().replace('\n', '') for x in imagenet_classes
    ]
    imagenet_classes = [key_to_classname[x] for x in abstract_imagenet_classes]

    print('\n')
    iterator = tqdm.tqdm(dataloader, 'Getting ImageNet pseudolabels...')
    memory_collect = []
    train_labels = []
    class_embed_collect = {}
    sample_reassign_topk = []

    for i, data_input in enumerate(iterator):
        with torch.no_grad():
            input = data_input[1]['image']
            out = model(input.to(device))
        for idx, label in zip(out, data_input[0].cpu().detach().numpy()):
            if label not in class_embed_collect:
                class_embed_collect[label] = []
            class_embed_collect[label].append(idx.detach().cpu().numpy())
        train_labels.extend(data_input[0].cpu().detach().numpy().tolist())
        sample_reassign_topk.extend(
            np.array(imagenet_classes)[np.argsort(
                out.detach().cpu().numpy(), axis=1)[:,
                                                    -topk:][:, ::-1]].tolist())

    class_collect_topk = {
        key: np.argsort(np.stack(item, axis=0).mean(axis=0))[-topk:][::-1]
        for key, item in class_embed_collect.items()
    }

    label_reassign_topk = [[] for _ in range(topk)]
    for k in range(topk):
        for label in np.unique(train_labels):
            label_reassign_topk[k].append(
                imagenet_classes[class_collect_topk[label][k]])
    if not full_label:
        label_reassign_topk = [[x.split(', ')[0] for x in y]
                               for y in label_reassign_topk]
        sample_reassign_topk = [[x.split(', ')[0] for x in y]
                                for y in sample_reassign_topk]

    if was_training:
        _ = model.train()

    dataloader.dataset.crop_size = crop_size
    dataloader.dataset.base_size = base_size
    dataloader.dataset.provide_transforms()

    return label_reassign_topk, sample_reassign_topk