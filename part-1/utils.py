import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)

QWERTY_NEIGHBORS = {
    'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'ersfxc', 'e': 'rdsw',
    'f': 'rtgdcv', 'g': 'tyhfvb', 'h': 'yugjbn', 'i': 'uokj', 'j': 'uihknm',
    'k': 'iojlm', 'l': 'opk', 'm': 'jkn', 'n': 'bhjm', 'o': 'iplk',
    'p': 'ol', 'q': 'wa', 'r': 'etdf', 's': 'wedxza', 't': 'ryfg',
    'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tugh',
    'z': 'asx',
}

def get_synonym(word):
    synsets = wordnet.synsets(word)
    synonyms_set = set()
    for synonym in synsets:
        for lemma in synonym.lemmas():
            candidate = lemma.name().replace('_', ' ')
            if candidate.lower() != word.lower():
                synonyms_set.add(candidate)
    return random.choice(list(synonyms_set)) if synonyms_set else word
 
def inject_typo(word):
    if len(word) < 3:
        return word
    eligible_words = [i for i, c in enumerate(word) if c.lower() in QWERTY_NEIGHBORS]

    if not eligible_words:
        return word
    idx = random.choice(eligible_words)
    c = word[idx].lower()
    replacement = random.choice(QWERTY_NEIGHBORS[c])
    return word[:idx] + replacement + word[idx + 1:]

def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    TYPO_PROBABILITY = 0.1
    SYNONYM_PROBABILITY = 0.2
 
    words = word_tokenize(example["text"])
    new_words = []
 
    for word in words:
        if word.isalpha() and len(word) > 2 and random.random() < SYNONYM_PROBABILITY:
            word = get_synonym(word)
 
        if len(word) >= 3 and random.random() < TYPO_PROBABILITY:
            word = inject_typo(word)
 
        new_words.append(word)
 
    example["text"] = TreebankWordDetokenizer().detokenize(new_words)

    # raise NotImplementedError

    ##### YOUR CODE ENDS HERE ######

    return example
