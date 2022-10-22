from surprise import Dataset, KNNBaseline, SVD, accuracy, Reader
a = KNNBaseline().fit().compute_similarities()
from surprise.similarities import cosine