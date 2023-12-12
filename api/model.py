from main import *


def model_job(name_dealer_product):
    model = DistanceRecommender(
        vectorizer=InfloatVectorizer(),
        simularity_func=cosine_similarity,
        text_prep_func=string_filter_emb
    )
    model.from_pretrained()
    return model.recommend(name_dealer_product)
