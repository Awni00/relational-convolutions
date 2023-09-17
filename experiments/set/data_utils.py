import numpy as np
import itertools
from setGame import SetGame

setgame = SetGame(())
all_cards = [(i,j) for i in range(9) for j in range(9)]

def contains_set(cards):
    return any(setgame.triple_is_set(triple) for triple in itertools.combinations(cards, 3))

def sample_nonset(k):
    idxs = np.random.choice(len(all_cards), k)
    candidate_tuple = [all_cards[i] for i in idxs]
    while contains_set(candidate_tuple):
        idxs = np.random.choice(len(all_cards), k)
        candidate_tuple = [all_cards[i] for i in idxs]
    return candidate_tuple

def sample_set(k, set_triples):
    # sample set triple
    set_cards = set_triples[np.random.choice(len(set_triples))]
    remaining_cards = [card for card in all_cards if card not in set_cards]
    idxs = np.random.choice(len(remaining_cards), k-3, replace=False)
    card_tuple = set_cards + [remaining_cards[i] for i in idxs]
    np.random.shuffle(card_tuple)
    return card_tuple

def create_set_classification_dataset(num_seqs, k, set_triples, card_embedder):

    vocab_size = 81
    setgame = SetGame()
    dim = len(card_embedder(np.expand_dims(setgame.image_of_card(0, 0), axis=0)).numpy().squeeze())

    # get embedding for each card
    card_images = np.zeros((9, 9, dim))
    for i in range(9):
        for j in range(9):
            card_images[i,j] = card_embedder(np.expand_dims(setgame.image_of_card(i, j), axis=0)).numpy().squeeze()

    object_seqs = np.zeros((num_seqs, k, dim))
    card_seqs = np.zeros((num_seqs, k, 2), dtype=int)
    labels = np.zeros(num_seqs, dtype=int)

    # sample tuples containing sets
    set_tuples = [sample_set(k, set_triples) for _ in range(num_seqs//2)]
    nonset_tuples = [sample_nonset(k) for _ in range(num_seqs//2)]

    # sample tuples not containing set

    # get card image embedding for each and create object_seqs, card_seqs, etc

    for s in np.arange(0, num_seqs, 2):
        for i in np.arange(k):
            card = set_tuples[s//2][i]
            object_seqs[s, i] = card_images[card[0], card[1]]
            card_seqs[s, i] = [card[0], card[1]]
        labels[s] = 1
        for i in np.arange(k):
            card = nonset_tuples[s//2][i]
            object_seqs[s+1, i] = card_images[card[0], card[1]]
            card_seqs[s+1, i] = [card[0], card[1]]
        labels[s+1] = 0

    return card_images, card_seqs, labels, object_seqs