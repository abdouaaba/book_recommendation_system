import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.recommendation_service import RecommendationService
from app.models.book import BookRecommendationRequest
from app.services.book_service import BookService
from app.core.config import settings

def calculate_diversity(recommendations):
    """Calculate diversity of recommendations based on unique titles."""
    return len(set(rec.title for rec in recommendations)) / len(recommendations)

def calculate_serendipity(recommendations, user_history):
    """Calculate serendipity based on how different recommendations are from user history."""
    rec_set = set(rec.title for rec in recommendations)
    history_set = set(user_history)
    return len(rec_set - history_set) / len(rec_set) if rec_set else 0

def evaluate_recommendations_first(recommendation_service, test_set, k=5):
    """Evaluate recommendations for a test set."""
    results = []
    for _, row in tqdm(test_set.iterrows(), total=len(test_set)):
        description = row['description']
        actual_title = row['title']
        
        faiss_recommendations = recommendation_service.recommend_books_faiss(description, k)
        cosine_recommendations = recommendation_service.recommend_books_cosine(description, k)
        
        faiss_titles = [rec.title for rec in faiss_recommendations]
        cosine_titles = [rec.title for rec in cosine_recommendations]
        
        results.append({
            'actual_title': actual_title,
            'faiss_hit': actual_title in faiss_titles,
            'cosine_hit': actual_title in cosine_titles,
            'faiss_diversity': calculate_diversity(faiss_recommendations),
            'cosine_diversity': calculate_diversity(cosine_recommendations),
            'faiss_serendipity': calculate_serendipity(faiss_recommendations, [actual_title]),
            'cosine_serendipity': calculate_serendipity(cosine_recommendations, [actual_title])
        })
    
    return pd.DataFrame(results)


def evaluate_recommendations_second(recommendation_service, test_queries, true_recommendations):
    faiss_precisions = []
    faiss_recalls = []
    faiss_mrrs = []

    cosine_precisions = []
    cosine_recalls = []
    cosine_mrrs = []

    for query, true_books in zip(test_queries, true_recommendations):
        request = BookRecommendationRequest(description=query, num_recommendations=3)
        
        faiss_recs = recommendation_service.recommend_books_faiss(request.description, request.num_recommendations)
        cosine_recs = recommendation_service.recommend_books_cosine(request.description, request.num_recommendations)
        
        faiss_titles = [rec.title for rec in faiss_recs]
        cosine_titles = [rec.title for rec in cosine_recs]
        
        true_titles = [book['title'] for book in true_books]
        
        faiss_precisions.append(calculate_precision(faiss_titles, true_titles))
        faiss_recalls.append(calculate_recall(faiss_titles, true_titles))
        faiss_mrrs.append(calculate_mrr(faiss_titles, true_titles))
        
        cosine_precisions.append(calculate_precision(cosine_titles, true_titles))
        cosine_recalls.append(calculate_recall(cosine_titles, true_titles))
        cosine_mrrs.append(calculate_mrr(cosine_titles, true_titles))
    
    results = {
        'faiss_precision': np.mean(faiss_precisions),
        'faiss_recall': np.mean(faiss_recalls),
        'faiss_mrr': np.mean(faiss_mrrs),
        'cosine_precision': np.mean(cosine_precisions),
        'cosine_recall': np.mean(cosine_recalls),
        'cosine_mrr': np.mean(cosine_mrrs),
    }
    
    return results

def calculate_precision(predicted_titles, true_titles):
    tp = 0
    for title in predicted_titles:
        if title in true_titles:
            tp += 1
    return tp / len(predicted_titles) if predicted_titles else 0

def calculate_recall(predicted_titles, true_titles):
    tp = 0
    for title in predicted_titles:
        if title in true_titles:
            tp += 1
    return tp / len(true_titles) if true_titles else 0

def calculate_mrr(predicted_titles, true_titles):
    for i, title in enumerate(predicted_titles):
        if title in true_titles:
            return 1 / (i + 1)
    return 0
    

def main():
    book_service = BookService()
    recommendation_service = RecommendationService()

    ### Hit Rate, Diversity, Serendipity
    
    # Load the full dataset
    df = book_service.load_dataframe()
    
    # Split the data into train and test sets
    _, test_set = train_test_split(df, test_size=0.2, random_state=42)
    
    # Evaluate recommendations
    results = evaluate_recommendations_first(recommendation_service, test_set)
    
    # Calculate and print metrics
    print("\nEvaluation Results:")
    print(f"FAISS Hit Rate: {results['faiss_hit'].mean():.2f}")
    print(f"Cosine Hit Rate: {results['cosine_hit'].mean():.2f}")
    print(f"FAISS Average Diversity: {results['faiss_diversity'].mean():.2f}")
    print(f"Cosine Average Diversity: {results['cosine_diversity'].mean():.2f}")
    print(f"FAISS Average Serendipity: {results['faiss_serendipity'].mean():.2f}")
    print(f"Cosine Average Serendipity: {results['cosine_serendipity'].mean():.2f}")
    

    # Suggestions for improvement based on evaluation results
    print("\nSuggestions for Improvement:")
    if results['faiss_hit'].mean() > results['cosine_hit'].mean():
        print("- FAISS outperforms Cosine similarity.")
    else:
        print("- Cosine similarity outperforms FAISS.")
    
    if results['faiss_diversity'].mean() < 0.8 or results['cosine_diversity'].mean() < 0.8:
        print("- Diversity of recommendations is low.")
    
    if results['faiss_serendipity'].mean() < 0.5 or results['cosine_serendipity'].mean() < 0.5:
        print("- Serendipity of recommendations is low.")


    ## Precision, Recall, MRR

    # Test queries and their corresponding expected recommendations
    test_queries = [
        "A tale of a young wizard discovering his magical heritage.",
        "A historical account of the events leading to World War II.",
        "A comprehensive guide to modern web development practices.",
    ]

    true_recommendations = [
        [
            {'id': 'book1', 'title': 'Wonders of the Invisible World - '},
            {'id': 'book2', 'title': 'Exploring the Fantasy Realm - Magic in Stories for Children and Young Adults'},
            {'id': 'book3', 'title': 'Immortaland - The Greatest Fantasy Kingdom To Exist And That Will Ever Exist'},
        ],
        [
            {'id': 'book4', 'title': 'A World at Arms - A Global History of World War II'},
            {'id': 'book5', 'title': 'The Third Reich in History and Memory - '},
            {'id': 'book6', 'title': 'All the People - '},
        ],
        [
            {'id': 'book7', 'title': 'Web Development Recipes - '},
            {'id': 'book8', 'title': 'The Modern Web - Multi-Device Web Development with HTML5, CSS3, and JavaScript'},
            {'id': 'book9', 'title': 'Practical Web Development - '},
        ],
    ]

    results = evaluate_recommendations_second(recommendation_service, test_queries, true_recommendations)
    
    print("Evaluation Results:")
    print(f"FAISS Precision: {results['faiss_precision']:.2f}")
    print(f"FAISS Recall: {results['faiss_recall']:.2f}")
    print(f"FAISS MRR: {results['faiss_mrr']:.2f}")
    print(f"Cosine Precision: {results['cosine_precision']:.2f}")
    print(f"Cosine Recall: {results['cosine_recall']:.2f}")
    print(f"Cosine MRR: {results['cosine_mrr']:.2f}")



if __name__ == "__main__":
    main()