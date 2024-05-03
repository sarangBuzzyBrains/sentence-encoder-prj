import tensorflow_hub as hub
import numpy as np

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print ("module %s loaded" % module_url)

def embed(input):
    return model(input)


def plot_similarity(m1_feat, m2_feat):
    corr = np.inner(m1_feat, m2_feat)
    return corr
   

def run_and_plot(message1, message2):
    m1_feat = embed(message1)
    m2_feat = embed(message2)
    res_matrix = plot_similarity(m1_feat, m2_feat)
    return res_matrix

