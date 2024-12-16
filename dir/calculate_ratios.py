import pandas as pd
import jellyfish
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, cityblock
from scipy.stats import pearsonr

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(name):
    """Compute embedding for a given name."""
    return model.encode(name)

# Function to compute all similarity metrics
def compute_similarity_metrics(row):
    name1 = str(row['name1']).strip().lower()
    name2 = str(row['name2']).strip().lower()

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

    # Return all computed metrics as a Series
    return pd.Series([soundex_ratio, metaphone_ratio, levenshtein_ratio, jaro_winkler_ratio,
                      cosine_sim, euclidean_sim, manhattan_sim, pearson_sim])

def calculate_ratios(input_csv, output_csv=None):
    # Load the input CSV file
    df = pd.read_csv(input_csv)
    output_csv="db/data1.csv"
    # Validate required columns
    if 'name1' not in df.columns or 'name2' not in df.columns:
        print("Error: Input CSV must contain 'name1' and 'name2' columns.")
        return
    # Compute similarity metrics for each row
    df[['soundex_ratio', 'metaphone_ratio', 'levenshtein_ratio', 'jaro_winkler_ratio',
        'cosine_similarity', 'euclidean_similarity', 'manhattan_similarity', 'pearson_similarity']] = df.apply(compute_similarity_metrics, axis=1)
    # Add an empty column with the name 'label' at the end of the DataFrame
    df['label'] = ''

    # Save the resulting DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)

    print(f"Output CSV file with similarity metrics saved at {output_csv}")
    return output_csv

# # Example usage
# if __name__ == "__main__":
#     input_csv_path = input("Enter the path to the input CSV file (with columns name1 and name2): ").strip()
#     output_csv_path = input("Enter the path to save the output CSV file: ").strip()
#     main(input_csv_path, output_csv_path)
