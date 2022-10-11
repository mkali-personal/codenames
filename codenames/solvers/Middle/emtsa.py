import random

from gensim.models import KeyedVectors
from numpy.linalg import norm

model = KeyedVectors.load_word2vec_format("language_data/english/wiki-50.bin", binary=True)
# %%
a, b = "war", "conflict"
a_vec, b_vec = model[a], model[b]
a_vec_normed, b_vec_normed = a_vec / norm(a_vec), b_vec / norm(b_vec)
mean_vec = (a_vec_normed + b_vec_normed) / 2
print(model.similar_by_vector(mean_vec))
print(model.most_similar(a, b))
print(model.most_similar_cosmul(a, b))
# %%
l = model.index_to_key[500:10000]
a = random.choice(l)
history_words = []


def input_a_word(history_words, first_turn=False):
    if first_turn:
        message = "choose initial random word: "
    else:
        message = "Choose middle word: "
    b = input(message)
    word_is_ok = b in model.index_to_key and b not in history_words
    while not word_is_ok:
        b = input("Your word is unknown - please try another one")
        word_is_ok = b in model.index_to_key
    return b


b = input_a_word(history_words, first_turn=True)

history_words.extend([a, b])

print(f"Computer first word is: {a}.\n")
i = 1
game_not_ended = True
while game_not_ended:
    a_vec, b_vec = model[a], model[b]
    a_vec_normed, b_vec_normed = a_vec / norm(a_vec), b_vec / norm(b_vec)
    mean_vec = (a_vec_normed + b_vec_normed) / 2
    most_similar = model.similar_by_vector(mean_vec)  # model.most_similar((a, b), topn=100)
    most_similar_words = [x[0] for x in most_similar if x[0] not in history_words]

    a = most_similar_words[0]
    b = input_a_word(history_words)
    print(f"computer chose: {a}.\n")
    history_words.extend([a, b])
    i += 1
    if a == b:
        print(f"Success! game ended within {i} trials.")
        game_not_ended = False
