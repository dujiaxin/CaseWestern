import torch
from torch.autograd import Variable
from transformers import *
from transformers import BertTokenizer, BertModel, WEIGHTS_NAME, CONFIG_NAME
import os
import json
# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
from torch.nn.utils.rnn import pad_sequence


def fill_sentence(tokenized_ids):
    seq_lengths = torch.LongTensor(list(map(len, tokenized_ids)))
    seq_tensor = Variable(torch.zeros((len(tokenized_ids), 60))).long()
    for idx, (seq, seqlen) in enumerate(zip(tokenized_ids, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
    return seq_tensor


def evaluate():
    pass


# train the credibility classifier model
def main():
    # Parameters:
    lr = 1e-3
    max_grad_norm = 1.0
    num_total_steps = 1000
    num_warmup_steps = 100
    warmup_proportion = float(num_warmup_steps) / float(num_total_steps)  # 0.1
    output_model_dir = "./model/"
    # Load the data
    with open('./data/input.json','r') as f:
        trainloader = json.load(f)
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=output_model_dir, do_lower_case=True, do_basic_tokenize=True)
    # Load pre-trained model (weights)
    bertModel = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

    #paragraph_encoder = torch.nn.gru()
    #document_encoder = torch.nn.GRU(768, 300)
    model = torch.nn.Linear(768,2)
    # Loss function
    criterion = torch.nn.BCELoss()  # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    ### In Transformers, optimizer and schedules are splitted and instantiated like this:
    # optimizer = AdamW(document_encoder.parameters(), lr=lr,
    #                   correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps,
    #                                  t_total=num_total_steps)  # PyTorch scheduler
    ### and used like this:
    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    bertModel.eval()

    for epoch in range(4):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data['document'].lower().split('.')
            label = data['is_credible']

            # If you have a GPU, put everything on cuda

            #bertModel.to('cuda')
            with torch.no_grad():  # When embedding the sentence use BERT, we don't train the model.
                # tokenized_text = tokenizer.tokenize(inputs[1])
                # #print(tokenized_text)
                # # Convert token to vocabulary indices
                # indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                # # Convert inputs to PyTorch tensors
                # tokens_tensor = torch.tensor([indexed_tokens])
                # #print(tokens_tensor)
                # #tokens_tensor = tokens_tensor.to('cuda')
                # #outputs = bertModel(tokens_tensor)
                # #last_hidden_state = outputs[0]
                indexed_tokens = []
                for ii, sentence in enumerate(inputs, 1):
                    tokenized_text = tokenizer.tokenize(sentence)
                    # Convert token to vocabulary indices
                    indexed_tokens.append(tokenizer.convert_tokens_to_ids(tokenized_text))
                # Convert inputs to PyTorch tensors

                #tokens_tensor = tokens_tensor.to('cuda')
                padded = fill_sentence(indexed_tokens)
                outputs = bertModel(padded)

            #predicted_is_credible = document_encoder(outputs)
            doc_embedding = torch.mean(outputs[0], (0,1))
            predicted_is_credible = model(doc_embedding)
            # zero the parameter gradients
            # optimizer.zero_grad()

            # forward + backward + optimize

            loss = criterion(predicted_is_credible, torch.tensor([1]).type(torch.FloatTensor))
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(),
            #                               max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            # scheduler.step()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    # Step 1: Save a model, configuration and vocabulary that you have fine-tuned

    # If we have a distributed model, save only the encapsulated model
    # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
    model_to_save = model.module if hasattr(model, 'module') else model

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(output_model_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_model_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_model_dir)
    print('Finished Training')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
