# https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/
# https://distill.pub/2016/augmented-rnns/
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax


def calculate_attention_vector(applied_attention):
    return np.sum(applied_attention, axis=1)


def single_dot_attention_score(dec_hidden_state, enc_hidden_state):
    # return the dot product of the two vectors
    return np.dot(dec_hidden_state, enc_hidden_state)


def dot_attention_score(dec_hidden_state, annotations):
    # return the product of dec_hidden_state transpose and enc_hidden_states
    return np.matmul(np.transpose(dec_hidden_state), annotations)


def apply_attention_scores(attention_weights, annotations):
    # Multiple the annotations by their weights
    return attention_weights * annotations




if __name__ == '__main__':
    dec_hidden_state = [5, 1, 20]  # The first input to the scoring function is the hidden state of decoder
    # Let's visualize our decoder hidden state
    plt.figure(figsize=(1.5, 4.5))
    sns.heatmap(np.transpose(np.matrix(dec_hidden_state)), annot=True, cmap=sns.light_palette("purple", as_cmap=True), linewidths=1)
    plt.show()
    annotation = [3, 12, 45]  # e.g. Encoder hidden state
    # Let's visualize the single annotation
    plt.figure(figsize=(1.5, 4.5))
    sns.heatmap(np.transpose(np.matrix(annotation)), annot=True, cmap=sns.light_palette("orange", as_cmap=True), linewidths=1)
    plt.show()

    annotations = np.transpose([[3, 12, 45], [59, 2, 5], [1, 43, 5], [4, 3, 45.3]])

    # Let's visualize our annotation (each column is an annotation)
    ax = sns.heatmap(annotations, annot=True, cmap=sns.light_palette("orange", as_cmap=True), linewidths=1)

    attention_weights_raw = dot_attention_score(dec_hidden_state, annotations)
    attention_weights = softmax(attention_weights_raw)
    applied_attention = apply_attention_scores(attention_weights, annotations)
    # Let's visualize our annotations after applying attention to them
    ax = sns.heatmap(applied_attention, annot=True, cmap=sns.light_palette("orange", as_cmap=True), linewidths=1)

    attention_vector = calculate_attention_vector(applied_attention)

    plt.figure(figsize=(1.5, 4.5))
    sns.heatmap(np.transpose(np.matrix(attention_vector)), annot=True, cmap=sns.light_palette("Blue", as_cmap=True), linewidths=1)
    plt.show()
