from sentence_transformers import SentenceTransformer, util
import numpy as np

def semantic_similarity_score(sentences, chosen_model = "all-distilroberta-v1"):

    model = SentenceTransformer(chosen_model)
    model = model.to('cpu')
    #create embeddings
    embeddings = model.encode(sentences)
    # Compute cosine-similarities for each sentence with each other sentence
    cosine_scores = util.cos_sim(embeddings, embeddings)
    # Extract the upper triangular part of the matrix, excluding the diagonal, so to then have an avarege score
    upper_triangle_values = cosine_scores.numpy()[np.triu_indices(n=len(sentences), k=1)]
    average_score = np.mean(upper_triangle_values)

    return average_score