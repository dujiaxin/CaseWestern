import argparse
import torch
from torch.autograd import Variable
import torch.utils.data as Data
from transformers import *
from transformers import BertTokenizer, BertModel, WEIGHTS_NAME, CONFIG_NAME
import os
import json
import random
import numpy as np
# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
from torch.nn.utils.rnn import pad_sequence

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange


class Classifier(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc = torch.nn.Linear(args.embed_dim, args.num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, status):
        classes = self.fc(status)
        return self.fc(status)

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
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def fill_sentence(tokenized_ids):
    seq_lengths = torch.LongTensor(list(map(len, tokenized_ids)))
    seq_tensor = Variable(torch.zeros((len(tokenized_ids), 60))).long()
    for idx, (seq, seqlen) in enumerate(zip(tokenized_ids, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
    return seq_tensor


def evaluate():
    pass

def read_data(filepath, epoch = 4):
    # print("read data")
    loader = None
    with open(filepath, 'r') as f:
        train_data = json.load(f)
        batch_size = int((len(train_data) / epoch)) + 1
        train_data = CWData(train_data)
        loader = Data.DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2, # number of process
        )
    return loader

# train the credibility classifier model
def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="input json for training. E.g., input.json")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    # Other parameters
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument('--seed', type=int, default=1,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=4,
                        help="epoch")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument("--learning_rate", default=0.03, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--transformer_dir", default='./model/', type=str,
                        help="The hugging face transformer cache directory.")
    # Load Parameters:
    args = parser.parse_args()
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    args.embed_dim = 768
    args.num_class = 1
    # Setup CUDA, GPU & distributed training
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.distributed.init_process_group(backend='nccl')
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    # Load the data
    # with open(args.train_file, 'r') as f:
    #     trainloader = json.load(f)
    trainloader = read_data(args.train_file, args.epoch)

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=args.transformer_dir, do_lower_case=True,
                                              do_basic_tokenize=True)
    # Load pre-trained model (weights)
    bertModel = BertModel.from_pretrained('bert-base-uncased', cache_dir=args.transformer_dir, output_hidden_states=True)

    # paragraph_encoder = torch.nn.gru()
    # document_encoder = torch.nn.GRU(768, 300)
    model = Classifier(args)
    # Loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    bertModel.eval()
    # If you have a GPU, put everything on cuda
    bertModel.to(args.device)
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    for epoch in range(args.epoch):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data['document'][i].lower().split('.')
            if data['credible_issue'][i]:
                label = 1
            else:
                label = 0
            sentence_embedding = torch.empty(768).to(args.device)
            with torch.no_grad():  # When embedding the sentence use BERT, we don't train the model.
                for ii, sentence in enumerate(inputs, 1):
                    tokenized_text = tokenizer.tokenize(sentence)
                    # Convert token to vocabulary indices
                    indexed_tokens = torch.tensor(tokenizer.convert_tokens_to_ids(tokenized_text))
                    outputs = bertModel(indexed_tokens.to(args.device).unsqueeze(0))
                    last4 = outputs[2][-4:]
                    last4_mean = torch.mean(torch.stack(last4), 0)  # [number_of_tokens, 768]
                    last4_mean = torch.mean(last4_mean.squeeze(0), 0)  # calculate the mean vector of each word
                    torch.stack((sentence_embedding, last4_mean),0)
            predicted_is_credible = model(sentence_embedding)
            # zero the parameter gradients
            # optimizer.zero_grad()

            # forward + backward + optimize
            loss = criterion(predicted_is_credible.unsqueeze(0),
                             torch.tensor([label]).unsqueeze(0).type(torch.FloatTensor).to(args.device))
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(),
            #                               max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            # scheduler.step()
            optimizer.step()

            # print statistics
            running_loss += loss
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

            # Step 1: Save a model, configuration and vocabulary that you have fine-tuned
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


if __name__ == '__main__':
    main()
