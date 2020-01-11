import argparse
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from transformers import *
import os
import json
import random
import numpy as np
import glob
# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
import re

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    # "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    # "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    # "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    # "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer),
    # "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
}


class Classifier(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.2)
        self.lstm = torch.nn.LSTM(args.embed_dim, args.lstm_hidden_dim, bidirectional=True, batch_first=True)
        self.fc = torch.nn.Linear(args.embed_dim * 2, args.num_class)
        # self.fc1 = torch.nn.Linear(args.embed_dim * 2, args.embed_dim * 4)
        # self.fc2 = torch.nn.Linear(args.embed_dim * 4, args.num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, status):
        lstmout = self.lstm(self.dropout(status))
        # lstmout = self.lstm(status)
        fcin = torch.cat([lstmout[1][1][0, :, :], lstmout[1][1][1, :, :]], dim=1)
        if torch.isnan(fcin).any():
            print('fcin is nan')
            print(lstmout)
        classes = self.fc(fcin.squeeze(0))
        return classes


class CWData(Data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def split_sentence(string, maxsplit=0):
    # split document to sentences
    # maybe can use nltk or better method
    delimiters = ".", "!", "-", ",", "\"", ";", "\n", "\t"
    regexPattern = '|'.join(map(re.escape, delimiters))
    return re.split(regexPattern, string, maxsplit)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def read_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    # random.shuffle(train_data)
    return train_data
    #     #batch_size = int((len(train_data) / args.epoch)) + 1
    #     batch_size = len(train_data)
    #     train_data = CWData(train_data)
    #     loader = Data.DataLoader(
    #         dataset=train_data,
    #         batch_size=batch_size,
    #         shuffle=True,
    #         num_workers=0, # number of process
    #     )
    # return loader


def train(args, model, bertModel, tokenizer, criterion):
    """ Train the model """

    # Optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    # optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.epoch
    # )

    bertModel.eval()
    logger.info("Training/evaluation parameters %s", args)
    for epoch in range(args.epoch):
        print('now in epoch ' + str(epoch))
        # Load the data
        trainloader = read_data(args.train_file)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs = data['document'].replace('\n', ' ').lower().split('.')
            inputs = split_sentence(data['document'])
            if data['credible_issue']:
                label = 1
            else:
                label = 0
            sentence_embedding = torch.zeros(args.embed_dim).to(args.device).unsqueeze(0)
            with torch.no_grad():  # When embedding the sentence use BERT, we don't train the model.
                for ii, sentence in enumerate(inputs, 2):
                    if len(sentence) < 3:
                        # print(sentence)
                        continue
                    elif len(sentence) > args.sentence_max_length:
                        print(data['rms'])
                        print(sentence)
                        continue
                    indexed_tokens = torch.tensor(
                        tokenizer.encode(sentence, add_special_tokens=True, max_length=args.sentence_max_length,
                                         pad_to_max_length=True)).unsqueeze(
                        0)  # Batch size 1
                    outputs = bertModel(indexed_tokens.to(args.device))
                    last_cls = outputs[0][:, 0, :]
                    sentence_embedding = torch.cat([sentence_embedding, last_cls], dim=0)
                    if torch.isnan(sentence_embedding).any():
                        print('sentence_embedding nan')
                        print(sentence_embedding)
                        print(data['rms'])
                        print(sentence)
            predicted_is_credible = model(sentence_embedding[1:].unsqueeze(0))
            # zero the parameter gradients

            # forward + backward + optimize
            optimizer.zero_grad()
            loss = criterion(predicted_is_credible.view(-1),
                             torch.tensor([label]).view(-1).type(torch.FloatTensor).to(args.device))
            if torch.isnan(loss).any():
                print('loss nan')
                print(i)
                print(data['rms'])

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(),
            #                               max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            # scheduler.step()
            optimizer.step()

            # print statistics
            running_loss += loss
            # print(i)
            if i % 200 == 199:  # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

            # Step 1: Save a model, configuration and vocabulary that you have fine-tuned
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(epoch))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            # If we have a distributed model, save only the encapsulated model
            # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
    print('Finished Training')


def evaluate(args, model, bertModel, tokenizer, criterion):
    print('evaluate in progress')
    trainloader = read_data(args.eval_file)
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    bertModel.eval()
    model.eval()
    for i, data in enumerate(tqdm(trainloader), 0):
        # get the inputs; data is a list of [inputs, labels]
        # inputs = data['document'].replace('\n', ' ').lower().split('.')
        inputs = split_sentence(data['document'])
        if data['credible_issue']:
            label = 1
        else:
            label = 0
        sentence_embedding = torch.zeros(args.embed_dim).to(args.device).unsqueeze(0)

        with torch.no_grad():  # When embedding the sentence use BERT, we don't train the model.
            for ii, sentence in enumerate(inputs, 2):
                if len(sentence) < 3:
                    # print(sentence)
                    continue
                elif len(sentence) > args.sentence_max_length:
                    print(data['rms'])
                    print(sentence)
                    continue
                indexed_tokens = torch.tensor(
                    tokenizer.encode(sentence, add_special_tokens=True, max_length=args.sentence_max_length,
                                     pad_to_max_length=True)).unsqueeze(
                    0)  # Batch size 1
                outputs = bertModel(indexed_tokens.to(args.device))
                last_cls = outputs[0][:, 0, :]
                if torch.isnan(last_cls).any():
                    print('bert nan')
                    print(last_cls)
                    print(data['rms'])
                    print(sentence)
                sentence_embedding = torch.cat([sentence_embedding, last_cls], dim=0)
        predicted_is_credible = model(sentence_embedding[1:].unsqueeze(0))
        # zero the parameter gradients
        # optimizer.zero_grad()

        # forward + backward + optimize
        loss2 = criterion(predicted_is_credible.view(-1),
                          torch.tensor([label]).view(-1).type(torch.FloatTensor).to(args.device))
        if label == 1 and loss2 < args.threshold:
            true_pos = true_pos + 1
        elif label == 0 and loss2 < args.threshold:
            true_neg = true_neg + 1
        elif label == 1 and loss2 > args.threshold:
            false_neg = false_neg + 1
        elif label == 0 and loss2 > args.threshold:
            false_pos = false_pos + 1
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    print("true_pos: " + str(true_pos))
    print("true_neg: " + str(true_neg))
    print("false_pos: " + str(false_pos))
    print("false_neg: " + str(false_neg))
    print("precision: " + str(precision))
    print("recall: " + str(recall))
    print('F1: ' + str(2 * precision * recall / (precision + recall)))


# train the credibility classifier model
def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="input json for training. E.g., train.json")
    parser.add_argument("--eval_file", default=None, type=str, required=True,
                        help="input json for evaluation. E.g., dev.json")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    # Other parameters
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument('--seed', type=int, default=1,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=1,
                        help="epoch")
    parser.add_argument('--lstm_hidden_dim', type=int, default=768,
                        help="lstm_hidden_dim in classifier")
    parser.add_argument("--embed_dim", type=int, default=768, help="LM model hidden size")
    parser.add_argument('--sentence_max_length', type=int, default=512,
                        help="sentence_max_length")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument("--threshold", default=0.5, type=float, help="classification threshold")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--learning_rate", default=1e-6, type=float,
                        help="The initial learning rate for Adam.")
    # parser.add_argument("--momentum", default=0.9, type=float,
    #                     help="The initial learning rate for SGD.")
    parser.add_argument("--transformer_dir", default='./model/', type=str,
                        help="The hugging face transformer cache directory.")
    parser.add_argument("--target_GPU", default='cuda', type=str,
                        help="The cuda you want to use. [cuda, cuda:0, cuda:1]")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    # Load Parameters:
    args = parser.parse_args()
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    args.num_class = 1
    args.do_lower_case = True
    # Setup CUDA, GPU & distributed training
    args.device = torch.device(args.target_GPU if torch.cuda.is_available() else "cpu")
    # torch.distributed.init_process_group(backend='nccl')
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    # Load pre-trained model tokenizer (vocabulary)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained('bert-base-uncased', cache_dir=args.transformer_dir, do_lower_case=True,
                                                do_basic_tokenize=True)
    # Load pre-trained model (weights)
    bertModel = model_class.from_pretrained('bert-base-uncased', cache_dir=args.transformer_dir,
                                            output_hidden_states=True)

    # paragraph_encoder = torch.nn.gru()
    # document_encoder = torch.nn.GRU(768, 300)
    model = Classifier(args)
    # Loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    # If you have a GPU, put everything on cuda
    bertModel.to(args.device)
    model.to(args.device)
    if args.do_train:
        train(args, model, bertModel, tokenizer, criterion)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            # If we have a distributed model, save only the encapsulated model
            # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        print('Finished Training')
    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            print(checkpoint)
            output_model_file = os.path.join(checkpoint, WEIGHTS_NAME)
            model.load_state_dict(torch.load(output_model_file))
            evaluate(args, model, bertModel, tokenizer, criterion)

    print('end program')
    return


if __name__ == '__main__':
    main()
