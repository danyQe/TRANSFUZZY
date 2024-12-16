import pandas as pd
import joblib
import numpy as np
import jellyfish
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, cityblock
from scipy.stats import pearsonr

def compute_similarity_metrics(name1, name2):
    """Compute similarity metrics for the given names."""
    # Check if the names start with the same character
    if name1[0].lower() != name2[0].lower():
        # If not, return a set of zeros for metrics and set the similarity result as False
        return 0, 0, 0, 0, 0, 0, 0, 0

    # Compute Soundex and Metaphone for both names
    soundex1 = jellyfish.soundex(name1) or 'N/A'
    metaphone1 = jellyfish.metaphone(name1) or 'N/A'
    soundex2 = jellyfish.soundex(name2) or 'N/A'
    metaphone2 = jellyfish.metaphone(name2) or 'N/A'

    # Calculate fuzzy ratios
    soundex_ratio = fuzz.ratio(soundex1, soundex2) / 100.0 if soundex1 and soundex2 else 0
    metaphone_ratio = fuzz.ratio(metaphone1, metaphone2) / 100.0 if metaphone1 and metaphone2 else 0

    # Calculate Levenshtein distance and ratio
    lev_distance = jellyfish.levenshtein_distance(name1, name2)
    max_len = max(len(name1), len(name2))
    levenshtein_ratio = 1 - (lev_distance / max_len) if max_len > 0 else 0

    # Calculate the Jaro-Winkler similarity score
    jaro_winkler_ratio = jellyfish.jaro_winkler_similarity(name1, name2)

    # Compute Embeddings for Cosine, Euclidean, Manhattan, and Pearson Similarity
    embedding1 = get_embedding(name1)
    embedding2 = get_embedding(name2)

    # Cosine Similarity
    cosine_sim = cosine_similarity([embedding1], [embedding2])[0][0]

    # Euclidean Similarity
    euclidean_dist = euclidean(embedding1, embedding2)
    euclidean_sim = 1 / (1 + euclidean_dist)  # Convert to similarity

    # Manhattan Similarity
    manhattan_dist = cityblock(embedding1, embedding2)
    manhattan_sim = 1 / (1 + manhattan_dist)  # Convert to similarity

    # Pearson Correlation Similarity
    pearson_corr, _ = pearsonr(embedding1, embedding2)
    pearson_sim = (pearson_corr + 1) / 2  # Normalize to [0, 1]

    # Return all computed metrics as a tuple
    return soundex_ratio, metaphone_ratio, levenshtein_ratio, jaro_winkler_ratio, cosine_sim, euclidean_sim, manhattan_sim, pearson_sim

# Function to get the embedding for a name using SentenceTransformer
def get_embedding(name):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(name)

def load_model(model_path):
    """Load the trained model from the specified path."""
    return joblib.load(model_path)

def preprocess_row(row):
    """Preprocess a single row of data in the same way as the training data."""
    row_data = row.drop(['name1', 'name2'])  # Drop name columns for model input
    return row_data.values.reshape(1, -1)  # Reshape for prediction

def predict_label(model, input_row):
    """Use the model to predict a label for the input row."""
    # Make sure input_row is a DataFrame with the correct feature names
    input_df = pd.DataFrame(input_row, columns=model.feature_names_in_)
    prediction = model.predict(input_df)
    return prediction[0]

def compare_names(input_file):
    # Load the trained model
    model = load_model('db/best_random_forest_model.pkl')
    if model is None:
        print("Error: Model not found at the specified path.")
        return

    # Load the dataset
    data = pd.read_csv(input_file)

    # Check for missing values and handle them
    if data.isnull().sum().any():
        print("Missing values detected. Filling with 0.")
        data.fillna(0, inplace=True)

    # Create a list to hold the predictions
    results = []
    for index, row in data.iterrows():
        # Skip rows where names don't start with the same character
        if row['name1'][0].lower() != row['name2'][0].lower():
            continue

        # Compute similarity metrics
        ratios = compute_similarity_metrics(row['name1'], row['name2'])

        # Create a DataFrame for prediction
        df = pd.DataFrame({
            'name1': [row['name1']],
            'name2': [row['name2']],
            'soundex_ratio': [ratios[0]],
            'metaphone_ratio': [ratios[1]],
            'levenshtein_ratio': [ratios[2]],
            'jaro_winkler_ratio': [ratios[3]],
            'cosine_sim': [ratios[4]],
            'euclidean_sim': [ratios[5]],
            'manhattan_sim': [ratios[6]],
            'pearson_sim': [ratios[7]]
        })

        # Preprocess the row for prediction
        input_row = preprocess_row(df.iloc[0])  # Get the first row as an array
        prediction = predict_label(model, input_row)  # Predict label

        # Append the results
        results.append({
            'name1': row['name1'],
            'name2': row['name2'],
            'predicted_label': prediction
        })

    # Create a DataFrame for the results
    results_df = pd.DataFrame(results)

    # Display the results
    print("Predictions for rows where names start with the same character:")
    print(results_df)

    # Optionally, save the predictions to a new CSV file
    results_df.to_csv("db/filtered_predictions2.csv", index=False)
    print("Filtered predictions saved as 'filtered_predictions2.csv'.")
    matched_names=[]    # Print name2 if the prediction value is 'y'
    for index, row in results_df.iterrows():
        if row['predicted_label'] == 'y':
            print("matched_names:",row['name2'])
            matched_names.append(row['name2'])
    return matched_names  
# compare_names("db/data1.csv")  
# # Run the main function with the input file path
# if __name__ == "__main__":
#     input_file_path = input("Enter the path to the CSV file for predictions: ")
#     main(input_file_path)
