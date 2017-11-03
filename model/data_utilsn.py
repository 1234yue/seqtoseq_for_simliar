# encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gzip
import os
import io
import re
import tarfile
import fileinput
from tensorflow.python.platform import gfile
from six.moves import urllib
import jieba
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile("\d")
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
def process_sentence(line):
    if 'question:' in line:
        seg_list = jieba.cut(line[9:-1], cut_all=False)
        return " ".join(seg_list)
    elif 'answer:' in line:
        seg_list = jieba.cut(line[7:-1], cut_all=False)
        return " ".join(seg_list)
    return None
def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.append(space_separated_fragment)
  return [w for w in words if w]
def process_data(source_path,target_path):
    with io.open(source_path,mode ="r") as f:
        with io.open(target_path,mode ="w",encoding='utf8') as fw:
            splitt=''
            for line in f.readlines():
                sen = process_sentence(line)
                if sen:
                    fw.write(process_sentence(line)+'\n')
def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size=100000,
                      tokenizer=None, normalize_digits=True):
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with io.open(data_path , mode='r',encoding='utf8') as f:
            index = 0
            for line in f.readlines():
                index = index + 1
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for word in tokens:
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = sorted(vocab, key=vocab.get, reverse=True)
            print(len(vocab_list))
            if len(vocab_list) > max_vocabulary_size:
                vocab_list =_START_VOCAB+ vocab_list[:max_vocabulary_size]
            with io.open(vocabulary_path,mode ="w",encoding='utf8') as fw:
                for w in vocab_list:
                    fw.write(w + "\n")
###
def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with open(vocabulary_path,mode='r',encoding='utf-8') as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, 3) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(_DIGIT_RE.sub("0", w), UNK_ID) for w in words]

def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with io.open(data_path, mode="r",encoding='utf8') as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab,
                                            tokenizer, normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")
###
def id_to_data(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    _,vocab = initialize_vocabulary(vocabulary_path)
    f = open(data_path, mode='w', encoding='utf-8')
    with gfile.GFile(target_path, mode="r") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = line.split()
          f.write(" ".join([vocab[int(tok)] for tok in token_ids]) + "\n")
##

if __name__ =='__main__':
    source_path ='/home/yueshifeng/repos/yuliao/mafengwo.txt'
    target_path ='/home/yuanfengcheng/repos_yue/yuliao/yuliao.txt'
    cidian_path ='/home/yuanfengcheng/repos_yue/yuliao/dict.in'
    data_path ='/home/yuanfengcheng/repos_yue/yuliao/chat.in'
    create_vocabulary(cidian_path,target_path)
    #data_to_token_ids(target_path,data_path,cidian_path)
