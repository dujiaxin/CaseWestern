import torch
import torchtext.vocab as vocab
from tqdm import tqdm
import numpy as np
import json
import nltk
import random
import argparse
import torch
from torch.autograd import Variable
import os
import json
import random
import numpy as np



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_word(glove, word):
    return glove.vectors[glove.stoi[word]]


def closest(glove, vec, n=10):
    """
    Find the closest words for a given vector
    """
    all_dists = [(w, torch.dist(vec, get_word(w))) for w in glove.itos]
    return sorted(all_dists, key=lambda t: t[1])[:n]


def print_tuples(tuples):
    for tuple in tuples:
        print('(%.4f) %s' % (tuple[1], tuple[0]))


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
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument("--learning_rate", default=0.03, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="The initial learning rate for SGD.")

    # Load Parameters:
    args = parser.parse_args()
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    args.num_class = 1
    args.epoch = 1
    # Setup CUDA, GPU & distributed training
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.distributed.init_process_group(backend='nccl')
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    # Load the data
    glove = vocab.GloVe(name='840B', dim=300)
    print('Loaded {} words'.format(len(glove.itos)))
    model = Classifier(args)
    # Loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    # If you have a GPU, put everything on cuda
    model.to(args.device)

    for epoch in range(args.epoch):
        print('new epoch')
        running_loss = 0.0
        with open(args.train_file, 'r') as f:
            trainloader = json.load(f)
            for i, items in enumerate(tqdm(trainloader)):
                tokens = nltk.word_tokenize(items['document'].lower())
                word_embeddings = torch.empty(300).to(args.device)
                for ii, word in enumerate(tokens, 10):
                    word_embeddings = torch.concat(word_embeddings, get_word(glove, word), dim=0)
                predicted_is_credible = model(word_embeddings)
                if items['credible_issue']:
                    label = torch.tensor(1).type(torch.FloatTensor).to(args.device)
                else:
                    label = torch.tensor(0).type(torch.FloatTensor).to(args.device)

            # zero the parameter gradients
            # optimizer.zero_grad()

            # forward + backward + optimize
            loss = criterion(predicted_is_credible.unsqueeze(0),
                             torch.tensor(label).type(torch.FloatTensor).to(args.device))
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
    print_tuples(closest(get_word('google')))