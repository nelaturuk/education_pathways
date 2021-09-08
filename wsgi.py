from __init__ import create_app
import pickle
import networkx as nx
import pandas as pd
from scipy.sparse import load_npz

if __name__=="__main__":
    app = create_app()
    with open('resources/course_vectorizer.pickle4','rb') as f:
        vectorizer = pickle.load(f)
    with open('resources/wordvecs.npz','rb') as f:
        out = load_npz(f)
    with open('resources/graph.pickle4','rb') as f:
        G = nx.read_gpickle(f)
    df = pd.read_pickle('resources/courses.pickle4').set_index('Code')
    app.run()
