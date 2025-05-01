# https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
import codecs
import csv
import itertools
import json
import math
import os
import random
import re
import unicodedata
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.jit import script, trace


# %%


def print_lines(filepath, n=10):
    with open(filepath, "rb") as f:
        lines = f.readlines()
    for line in lines[:n]:
        print(line)


corpus_path = "./data/raw/movie-corpus/utterances.jsonl"

print_lines(corpus_path)


# %%


def load_lines_and_conversations(file_path):
    lines, conversations = {}, {}
    with open(file_path, "r", encoding="iso-8859-1") as f:
        for line in f:
            line_json = json.loads(line)
            line_id = line_json["id"]
            convo_id = line_json["conversation_id"]

            line_obj = {
                "lineID": line_id,
                "characterID": line_json["speaker"],
                "text": line_json["text"],
            }
            lines[line_id] = line_obj

            if convo_id not in conversations:
                conv_obj = {
                    "conversationID": convo_id,
                    "movieID": line_json["meta"]["movie_id"],
                    "lines": [line_obj],
                }
            else:
                conv_obj = conversations[convo_id]
                conv_obj["lines"].insert(0, line_obj)

            conversations[convo_id] = conv_obj

    return lines, conversations


def extract_sentence_pairs(conversations):
    qa_pairs = []
    for _, conversation in conversations.items():
        n_lines = len(conversation["lines"])

        for i in range(n_lines - 1):  # ignore last line
            input_line = conversation["lines"][i]["text"].strip()
            target_line = conversation["lines"][i + 1]["text"].strip()

            if input_line and target_line:
                qa_pairs.append([input_line, target_line])

    return qa_pairs


# %%

processed_data_path = "./data/processed/formatted_movie_lines.txt"

delim = "\t"
delim = str(codecs.decode(delim, "unicode_escape"))

print("Processing corpus into lines and conversations..")
lines, conversations = load_lines_and_conversations(corpus_path)

print("Writing newly formatted file..")
with open(processed_data_path, "w", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter=delim, lineterminator="\n")
    for pair in extract_sentence_pairs(conversations):
        writer.writerow(pair)

print("Sample lines from file:")
print_lines(processed_data_path)


# %%

PAD_token = 0
SOS_token = 1
EOS_token = 2


class Vocab:
    def __init__(self, name):
        self.name = name
        self.trimmed = False

        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            PAD_token: "PAD",
            SOS_token: "SOS",
            EOS_token: "EOS",
        }
        self.num_words = len(self.index2word)

    def add_sentence(self, sentence):
        for word in sentence.split(" "):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            ind = self.num_words
            self.word2index[word] = ind
            self.word2count[word] = 1
            self.index2word[ind] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print(
            "keep_words {} / {} = {:.4f}".format(
                len(keep_words),
                len(self.word2index),
                len(keep_words) / len(self.word2index),
            )
        )

        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            PAD_token: "PAD",
            SOS_token: "SOS",
            EOS_token: "EOS",
        }
        self.num_words = len(self.index2word)

        for word in keep_words:
            self.add_word(word)

# %%

MAX_LENGTH = 10

def unicode_to_ascii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

def normalise_string(s):
    s = s.lower().strip()
    s = unicode_to_ascii(s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s)
    s = s.strip()
    return s

def read_vocs(data_path, corpus_name):
    print("Reading lines..")
    with open(data_path, encoding="utf-8") as f:
        lines = f.read().strip().split("\n")

    pairs = [[normalise_string(s) for s in l.split("\t")] for l in lines]
    voc = Vocab(corpus_name)
    return voc, pairs


def filter_pair(p):
    return len(p[0].split(" ")) < MAX_LENGTH and len(p[1].split(" ")) < MAX_LENGTH


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def load_prepare_data(corpus_name, data_path):
    print("Start preparing training data..")

    voc, pairs = read_vocs(data_path, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))

    pairs = filter_pairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))

    print("Counting words..")
    for p in pairs:
        # TODO investigate incidence of multiple tabs per line
        voc.add_sentence(p[0])
        voc.add_sentence(p[1])

    print("Counted words: ", voc.num_words)
    return voc, pairs

corpus_name = "movie-corpus"
voc, pairs = load_prepare_data(corpus_name, processed_data_path)

print("pairs")
for pair in pairs[:10]:
    print(pair)



# %%
# %%
# %%
# %%
# %%
# %%


