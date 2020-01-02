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
import glob
# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
from torch.nn.utils.rnn import pad_sequence

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    #"xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    #"xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    #"roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    #"distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer),
    #"xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
}

class Classifier(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.lstm = torch.nn.LSTM(args.embed_dim, args.lstm_hidden_dim, bidirectional=True, batch_first=True)
        self.fc = torch.nn.Linear(args.embed_dim*2, args.num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, status):
        status[status != status] = 0
        lstmout= self.lstm(status)
        fcin = torch.cat([lstmout[1][1][0,:,:], lstmout[1][1][1,:,:]], dim=1)
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
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def fill_sentence(tokenized_ids):
    seq_lengths = torch.LongTensor(list(map(len, tokenized_ids)))
    seq_tensor = Variable(torch.zeros((len(tokenized_ids), 60))).long()
    for idx, (seq, seqlen) in enumerate(zip(tokenized_ids, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
    return seq_tensor


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def read_data(args):
    # print("read data")
    loader = None
    with open(args.train_file, 'r') as f:
        train_data = json.load(f)
        #batch_size = int((len(train_data) / args.epoch)) + 1
        batch_size = len(train_data)
        train_data = CWData(train_data)
        loader = Data.DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0, # number of process
        )
    return loader


def train(args, model, bertModel, tokenizer, criterion):
    """ Train the model """
    # Load the data
    # with open(args.train_file, 'r') as f:
    #     trainloader = json.load(f)
    trainloader = read_data(args)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    logger.info("Training/evaluation parameters %s", args)
    for epoch in range(args.epoch):
        print('now in epoch ' + str(epoch))
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data['document'][i].replace('\n', ' ').lower().split('.')
            if data['credible_issue'][i]:
                label = 1
            else:
                label = 0
            sentence_embedding = torch.empty(768).to(args.device).unsqueeze(0)
            with torch.no_grad():  # When embedding the sentence use BERT, we don't train the model.
                for ii, sentence in enumerate(inputs, 2):
                    if len(sentence) < 3:
                        continue
                    indexed_tokens = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(
                        0)  # Batch size 1
                    outputs = bertModel(indexed_tokens.to(args.device))
                    last_cls = outputs[0][:, 0, :]
                    sentence_embedding = torch.cat([sentence_embedding, last_cls], dim=0)
            predicted_is_credible = model(sentence_embedding.unsqueeze(0))
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
            print(i)
            if i % 20 == 19:  # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
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


def evaluate(args, model, bertModel, tokenizer, criterion):
    trainloader = read_data(args)
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    model.load_state_dict(torch.load(output_model_file))
    model.eval()
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data['document'][i].lower().split('.')
        if data['credible_issue'][i]:
            label = 1
        else:
            label = 0
        sentence_embedding = torch.empty(768).to(args.device).unsqueeze(0)

        with torch.no_grad():  # When embedding the sentence use BERT, we don't train the model.
            for ii, sentence in enumerate(inputs, 1):
                if len(sentence) < 3:
                    print(sentence)
                    continue
                # Convert token to vocabulary indices
                indexed_tokens = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
                outputs = bertModel(indexed_tokens.to(args.device))
                last_cls = outputs[0][:,0,:]
                sentence_embedding = torch.cat([sentence_embedding, last_cls], dim=0)
            predicted_is_credible = model(sentence_embedding.unsqueeze(0))
        # zero the parameter gradients
        # optimizer.zero_grad()

        # forward + backward + optimize
            loss2 = criterion(predicted_is_credible.unsqueeze(0),
                             torch.tensor([label]).unsqueeze(0).type(torch.FloatTensor).to(args.device))
            if label == 1 and loss2 < 0.5:
                true_pos = true_pos+1
            elif label == 0 and loss2 < 0.5:
                true_neg = true_neg+1
            elif label == 1 and loss2 > 0.5:
                false_neg = false_neg+1
            elif label == 0 and loss2 > 0.5:
                false_pos = false_pos+1
    precision = true_pos/(true_pos+false_pos)
    recall = true_pos/(true_pos+false_neg)
    print("precision: "+str(precision))
    print("recall: " + str(recall))
    print('F1: ' +str(precision*recall/(precision+recall)))


# train the credibility classifier model
def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="input json for training. E.g., input.json")
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
    parser.add_argument('--epoch', type=int, default=4,
                        help="epoch")
    parser.add_argument('--lstm_hidden_dim', type=int, default=768,
                        help="lstm_hidden_dim in classifier")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument("--learning_rate", default=0.03, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--transformer_dir", default='./model/', type=str,
                        help="The hugging face transformer cache directory.")
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
    args.embed_dim = 768
    args.num_class = 1
    args.do_lower_case = True
    # Setup CUDA, GPU & distributed training
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.distributed.init_process_group(backend='nccl')
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)


    # Load pre-trained model tokenizer (vocabulary)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained('bert-base-uncased', cache_dir=args.transformer_dir, do_lower_case=True,
                                              do_basic_tokenize=True)
    # Load pre-trained model (weights)
    bertModel = model_class.from_pretrained('bert-base-uncased', cache_dir=args.transformer_dir, output_hidden_states=True)

    # paragraph_encoder = torch.nn.gru()
    # document_encoder = torch.nn.GRU(768, 300)
    model = Classifier(args)
    # Loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    bertModel.eval()
    # If you have a GPU, put everything on cuda
    bertModel.to(args.device)
    model.to(args.device)
    if args.do_train:
        train(args, model, bertModel, tokenizer, criterion)
    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            evaluate(args, model, bertModel, tokenizer, )

    print('end program')
    return



if __name__ == '__main__':
    main()
