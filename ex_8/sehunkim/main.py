import pickle
import numpy as np
import argparse
import os
from matplotlib import pyplot as plt
import seaborn as sns

def load_pickle_data(pickle_data):
    answer_models = np.array(pickle_data["answer_models"])
    outputs = np.array(pickle_data["output"])
    p_init = np.array(pickle_data["models"]["PI"])
    trans_mx = np.array(pickle_data["models"]["A"])
    out_mx = np.array(pickle_data["models"]["B"])

    return outputs, answer_models, p_init, trans_mx, out_mx


def forward(outputs, p_init, trans_mx, out_mx):
    """Function for generating labels with forward

    Args:
        outputs ([type]): [description]
        answer_model ([type]): [description]
        p_init ([type]): [description]
        trans_mx ([type]): [description]
        out_mx ([type]): [description]

    Returns:
        [type]: [description]
    """
    model_n, state_n, _ = p_init.shape
    series_n = outputs.shape[1]

    state_p = np.zeros((model_n, state_n, series_n))
    for i in range(model_n):
        state_p[i, :, 0] = p_init[i, :, 0]
        for j in range(1, series_n):
            state_p[i, :, j] = state_p[i, :, j-1] @ trans_mx[i]

    forward_label = np.zeros(series_n)
    for i, output in enumerate(outputs):
        likelihood = np.ones(model_n)
        for j, output_val in enumerate(output):
            for k in range(model_n):
                likelihood[k] *= state_p[k, :, j] @ out_mx[k, :, output_val]
        forward_label[i] = np.argmax(likelihood)

    return forward_label


def viterbi(outputs, p_init, trans_mx, out_mx):
    model_n, state_n, _ = p_init.shape
    series_n = outputs.shape[1]

    viterbi_label = np.zeros(series_n)
    for i, output in enumerate(outputs):
        state_p = np.zeros((model_n, state_n, series_n))
        for j in range(model_n):
            state_p[j, :, 0] = p_init[j, :, 0]
            for k, output_val in enumerate(output):
                if k == 0:
                    state_p[j, :, k] = state_p[j, :, k] * \
                        out_mx[j, :, output_val]
                else:
                    prob = (state_p[j, :, k - 1][:, np.newaxis] * trans_mx[j]
                            ).T * out_mx[j, :, output_val][:, np.newaxis]
                    state_p[j, :, k] = np.max(prob, axis=1)
        viterbi_label[i] = np.argmax(state_p[:, :, series_n-1]) // state_n

    return viterbi_label

def calc_accuracy(data1, data2):
    n_correct = 0
    if data1.shape != data2.shape:
        print("Error, shape does not match")
        exit()
    else:
        for i in range(len(data1)):
            if data1[i] == data2[i]:
                n_correct += 1
    accuracy = n_correct / len(data1)
    return accuracy
                
def heatmap(correct, predicted):
    cm = np.zeros((5,5))
    for i in range(len(correct)):
        cm[int(correct[i]),int(predicted[i])] += 1
    return cm

def main():
    parser = argparse.ArgumentParser(
        description='Program for forward and viterbi algorythm')
    parser.add_argument("-f", dest="filename", help='Filename', required=True)
    args = parser.parse_args()
    filename = os.path.splitext(args.filename)[0]
    
    with open(args.filename, "rb") as p:
        pickle_data = pickle.load(p)

    outputs, answer_models, p_init, trans_mx, out_mx = load_pickle_data(
        pickle_data)

    forward_results = forward(outputs, p_init, trans_mx, out_mx)
    viterbi_results = viterbi(outputs, p_init, trans_mx, out_mx)
    forward_accuracy = calc_accuracy(forward_results, answer_models)
    viterbi_accuracy = calc_accuracy(viterbi_results, answer_models)
    
    forward_hm = heatmap(answer_models, forward_results)
    viterbi_hm = heatmap(answer_models, viterbi_results)
    
    fig = plt.figure()
    
    plt.subplot(1,2,1)
    plt.title(f"Forward Algorithm for {args.filename}\n Accuracy = {forward_accuracy * 100}%")
    ax = sns.heatmap(forward_hm, annot = True, square = True,cmap = 'Greys' ,cbar = False, fmt = "1.1f")
    ax.set_xlabel('predicted')
    ax.set_ylabel('answer')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    
    plt.subplot(1,2,2)
    plt.title(f"Viterbi Algorithm for {args.filename}\n Accuracy = {viterbi_accuracy * 100}%")
    ax = sns.heatmap(viterbi_hm, annot = True, square = True,cmap = 'Greys' ,cbar = False, fmt = "1.1f")
    ax.set_xlabel('predicted')
    ax.set_ylabel('answer')
    fig.tight_layout()
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    plt.subplots_adjust(left = .1, right = .9,wspace = .5)
    plt.savefig(f"result_{filename}.png")
    

if __name__ == "__main__":
    main()
