# Standard Library
import argparse
import datetime
import json
import os
import random

# Others
import nltk
import numpy as np
import torch
import torch.nn.functional as F
import torchtext.vocab as vocab
from torch.autograd import Variable
from tqdm import tqdm


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
        print("(%.4f) %s" % (tuple[1], tuple[0]))


class Classifier(torch.nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args
        D = 300  # word embedding dimentions
        C = 1
        Ci = 1
        Co = 100  # number of each kind of kernel
        Ks = [3, 4, 5]  # 'comma-separated kernel size to use for convolution'

        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = torch.nn.ModuleList([torch.nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        """
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        """
        self.dropout = torch.nn.Dropout(args.dropout)
        self.fc1 = torch.nn.Linear(len(Ks) * Co, C)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = x.unsqueeze(0)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        # print(x.size)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        #  print(x.size)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        # print(x.size)
        x = torch.cat(x, 1)
        # print(x.size)
        """
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        """
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = F.sigmoid(self.fc1(x))  # (N, C)
        return logit


# train the credibility classifier model
def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_file", default=None, type=str, required=True, help="input json for training. E.g., input.json",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--seed", type=int, default=1, help="random seed for initialization")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--learning_rate", default=0.03, type=float, help="The initial learning rate for SGD.",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, help="The initial learning rate for SGD.",
    )
    parser.add_argument("--dropout", default=0.5, type=float, help="The drop out ratio in CNN.")

    # Load Parameters:
    args = parser.parse_args()
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )
    args.num_class = 1
    args.epoch = 1
    # Setup CUDA, GPU & distributed training
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.distributed.init_process_group(backend='nccl')
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    # Load the data
    glove = vocab.GloVe(name="840B", dim=300)
    print("Loaded {} words".format(len(glove.itos)))
    model = Classifier(args)
    # Loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    # If you have a GPU, put everything on cuda
    model.to(args.device)

    for epoch in range(args.epoch):
        print("new epoch")
        running_loss = 0.0
        with open(args.train_file, "r") as f:
            trainloader = json.load(f)
            for i, items in enumerate(tqdm(trainloader)):
                tokens = nltk.word_tokenize(items["document"].lower())
                word_embeddings = torch.empty(300).to(args.device).unsqueeze(0)
                for ii, word in enumerate(tokens, 10):
                    try:
                        word_embedding = get_word(glove, word).to(args.device).unsqueeze(0)
                        # print(word_embeddings.size())
                        word_embeddings = torch.cat((word_embeddings, word_embedding), 0)
                    except KeyError:
                        continue
                predicted_is_credible = model(word_embeddings)

                print(predicted_is_credible)

                if items["is_credible"]:
                    label = torch.tensor(1).type(torch.FloatTensor).to(args.device)
                else:
                    label = torch.tensor(0).type(torch.FloatTensor).to(args.device)

            # zero the parameter gradients
            # optimizer.zero_grad()

            # forward + backward + optimize
            loss = criterion(predicted_is_credible.unsqueeze(0), label)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(),
            #                               max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            # scheduler.step()
            optimizer.step()

            # print statistics
            running_loss += loss
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

            # Step 1: Save a model, configuration and vocabulary that you have fine-tuned
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            # If we have a distributed model, save only the encapsulated model
            # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
        model_to_save = model.module if hasattr(model, "module") else model
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir + datetime.datetime.now().strftime("%s"))
        torch.save(model_to_save.state_dict(), output_model_file)
        print("Finished Training")


if __name__ == "__main__":
    main()
    # print_tuples(closest(get_word('google')))
