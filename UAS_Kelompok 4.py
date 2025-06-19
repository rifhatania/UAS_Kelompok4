# 1. Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 2. Import dataset
print("=== LOADING DATASETS ===")
books = pd.read_csv('Books.csv')
ratings = pd.read_csv('Books-Ratings.csv')
users = pd.read_csv('Users.csv')

print(f"Books dataset shape: {books.shape}")
print(f"Ratings dataset shape: {ratings.shape}")
print(f"Users dataset shape: {users.shape}")

# 3. Display initial dataset information
print("\n=== DATASET OVERVIEW ===")
print("\nBooks columns:", books.columns.tolist())
print("Ratings columns:", ratings.columns.tolist())
print("Users columns:", users.columns.tolist())

print("\n=== DATASET RATINGS ===")
print(ratings.head())
print(f"\nRatings info:")
print(ratings.info())

print("\n=== DATASET USERS ===")
print(users.head())
print(f"\nUsers info:")
print(users.info())

print("\n=== DATASET BOOKS ===")
print(books.head())

# 4. PREPROCESSING DATA - Users
print("\n PREPROCESSING DATA")
print("\nA. PREPROCESSING USERS DATASET")

# Cek missing values
print("\nMissing value users sebelum dihapus:")
missing_before = users.isnull().sum()
print(missing_before)

# Hapus missing values
initial_user_count = users.shape[0]
users_clean = users.dropna()
removed_na = initial_user_count - users_clean.shape[0]
print(f"Hapus {removed_na} users dengan missing values")

# Tangani outlier umur dengan Z-score
print(f"\nHapus outlier umur dengan Z-score")
print("Statistika umur asli:")
print(users_clean['Age'].describe())

# Hitung Z-score
from scipy.stats import zscore
age_z = zscore(users_clean['Age'])
threshold = 3

# Filter outliers
users_no_outliers = users_clean[np.abs(age_z) < threshold]
outliers_removed = users_clean.shape[0] - users_no_outliers.shape[0]
print(f"Hapus {outliers_removed} users dengan outlier umur (Z-score > {threshold})")

# Statistik setelah pembersihan
print("Statistik umur setelah pembersihan:")
print(users_no_outliers['Age'].describe())

# 5. PREPROCESSING - Books and Ratings datasets
print("\nB. PREPROCESSING BOOKS DATASET")
print("Missing value users sebelum dihapus:")
print(books.isnull().sum())

books_original_size = len(books)
books_clean = books.dropna(subset=['Book-Title', 'Book-Author', 'Publisher'])
print(f"Hapus {books_original_size - len(books_clean)} books dengan missing essential data")

print("\nC. PREPROCESSING RATINGS DATASET")
print("Missing values pada rating sebelum dibersihkan :")
print(ratings.isnull().sum())

ratings_original_size = len(ratings)
ratings_clean = ratings.dropna()
print(f"Hapus {ratings_original_size - len(ratings_clean)} ratings dengan missing values")

print(f"\nDistribusi rating):")
print(ratings_clean['Book-Rating'].value_counts().sort_index())
print(f"Rating range: {ratings_clean['Book-Rating'].min()} to {ratings_clean['Book-Rating'].max()}")

# 6. COMBINE DATASETS - Users first, then Books
print("\n COMBINING DATASETS")
print("Step 1: Merge Ratings dengan Users (to filter valid users)")
ratings_with_users = pd.merge(ratings_clean, users_no_outliers[['User-ID']], on='User-ID', how='inner')
print(f"Ratings setelah user filtering: {len(ratings_with_users)} (menghapus {len(ratings_clean) - len(ratings_with_users)} ratings dari user outliers)")

print("Step 2: Merge with Books")
complete_dataset = pd.merge(ratings_with_users, books_clean[['ISBN', 'Book-Title', 'Book-Author', 'Publisher']], on='ISBN', how='inner')
print(f"Dataset shape setelah: {complete_dataset.shape}")

# 7. ENCODING AND NORMALIZATION
print("\nD. ENCODING")
print("Text data will be processed for content-based filtering")

print("\nE. NORMALISASI")
# Normalize ratings to 0-1 scale for some calculations
complete_dataset['Book-Rating-Normalized'] = complete_dataset['Book-Rating'] / complete_dataset['Book-Rating'].max()
print("Added normalized ratings")

# 8. Filter buku dengan minimal 20 rating
MIN_BOOK_RATINGS = 20

print("\n=== FILTERING POPULAR BOOKS ===")
print(f"Filtering books with at least {MIN_BOOK_RATINGS} ratings...")

# Hitung jumlah rating untuk setiap judul buku
book_rating_counts = complete_dataset['Book-Title'].value_counts()

# Ambil judul buku yang jumlah rating-nya memenuhi batas minimal
popular_book_titles = book_rating_counts.loc[book_rating_counts >= MIN_BOOK_RATINGS].index.tolist()

# Filter dataset agar hanya menyertakan buku-buku populer
ratings_filtered = complete_dataset[complete_dataset['Book-Title'].isin(popular_book_titles)]

# Tampilkan ringkasan sebelum dan sesudah filtering
print(f"Jumlah buku sebelum filter: {book_rating_counts.size}")
print(f"Number of books after filtering: {len(popular_book_titles)}")
print(f"Number of ratings before filtering: {len(complete_dataset)}")
print(f"Number of ratings after filtering: {len(ratings_filtered)}")

# 9. Membuat user-item rating matrix
print("\nCREATING USER-ITEM MATRIX")
userRatings = ratings_filtered.pivot_table(
    index=['User-ID'],
    columns=['Book-Title'],
    values='Book-Rating',
    fill_value=0
)

print("User-item matrix shape:", userRatings.shape)

# 10. SIMILARITY METHODS
print("\n=== SIMILARITY METHODS ===")

# A. EUCLIDEAN DISTANCE
print("\nA. EUCLIDEAN SIMILARITY")
def calculate_euclidean_similarity(matrix):
    # Calculate euclidean distances and convert to similarity
    distances = euclidean_distances(matrix.T)  # Transpose for item-item similarity
    # Convert distance to similarity (higher similarity = lower distance)
    similarities = 1 / (1 + distances)
    return pd.DataFrame(similarities, index=matrix.columns, columns=matrix.columns)

euclidean_sim = calculate_euclidean_similarity(userRatings)
print("Euclidean similarity matrix shape:", euclidean_sim.shape)

# B. COSINE SIMILARITY
print("\nB. COSINE SIMILARITY")
def calculate_cosine_similarity(matrix):
    # Calculate cosine similarity
    cos_sim = cosine_similarity(matrix.T)  # Transpose for item-item similarity
    return pd.DataFrame(cos_sim, index=matrix.columns, columns=matrix.columns)

cosine_sim = calculate_cosine_similarity(userRatings)
print("Cosine similarity matrix shape:", cosine_sim.shape)

# C. PEARSON CORRELATION (Optimized)
print("\nC. PEARSON CORRELATION")
print("Computing Pearson correlation matrix... (this may take a moment)")

# Option 1: Reduce matrix size by sampling if too large
if userRatings.shape[1] > 1000:  # If more than 1000 books
    print(f"Large matrix detected ({userRatings.shape[1]} books). Sampling top 500 most rated books for faster computation...")
    # Get top 500 most rated books
    book_counts = (userRatings > 0).sum(axis=0).sort_values(ascending=False)
    top_books = book_counts.head(500).index
    userRatings_sample = userRatings[top_books]
    corrMatrix = userRatings_sample.corr(method='pearson', min_periods=5)
    print(f"Correlation matrix computed with {len(top_books)} books")
else:
    corrMatrix = userRatings.corr(method='pearson', min_periods=10)

# Alternative: Fast approximation methods
print("\n=== ALTERNATIVE FAST SIMILARITY METHODS ===")

# D. FAST PEARSON APPROXIMATION (using numpy)
print("\nD. FAST PEARSON APPROXIMATION")
def fast_pearson_similarity(matrix, top_n=500):
    """Fast Pearson correlation using numpy operations"""
    # Sample top N most rated books if matrix is large
    if matrix.shape[1] > top_n:
        book_counts = (matrix > 0).sum(axis=0).sort_values(ascending=False)
        top_books = book_counts.head(top_n).index
        matrix_sample = matrix[top_books].T
    else:
        matrix_sample = matrix.T

    # Convert to numpy for faster computation
    data = matrix_sample.values

    # Compute correlation using numpy
    corr_matrix = np.corrcoef(data)

    # Convert back to DataFrame
    return pd.DataFrame(corr_matrix, index=matrix_sample.index, columns=matrix_sample.index)

# Use fast method if original is too slow
if userRatings.shape[1] > 200:
    print(f"Using fast approximation for large matrix ({userRatings.shape[1]} books)")
    fast_corrMatrix = fast_pearson_similarity(userRatings, top_n=300)
    print("Fast correlation matrix shape:", fast_corrMatrix.shape)

    # Use fast matrix for recommendations if needed
    corrMatrix_to_use = fast_corrMatrix
else:
    corrMatrix_to_use = corrMatrix

print("Matrix ready for recommendations")

## 11. Content-Based Filtering (judul + author + publisher)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Ambil hanya 10.000 data buku (atau kurang dari itu)
books_sample = books.head(20000).copy()

# Gabungkan fitur konten
books_sample['combined_features'] = books_sample['Book-Title'].astype(str) + ' ' + books_sample['Book-Author'].astype(str) + ' ' + books_sample['Publisher'].astype(str)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books_sample['combined_features'])

# Hitung similarity antar buku
content_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# Index mapping
indices = pd.Series(books_sample.index, index=books_sample['Book-Title']).drop_duplicates()

# Fungsi rekomendasi buku mirip
def recommend_books(title, sim_matrix=content_sim_matrix):
    idx = indices.get(title)
    if idx is None:
        return f"Buku '{title}' tidak ditemukan."

    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    book_indices = [i[0] for i in sim_scores]
    return books_sample.iloc[book_indices][['Book-Title', 'Book-Author', 'Publisher']]


# Contoh: Rekomendasi buku mirip dengan "Peter Pan: The Original Story (Peter Pan)"
recommended_books = recommend_books("Peter Pan: The Original Story (Peter Pan)")
recommended_books

# 12. EVALUASI BERDASARKAN DATA USER
print("\n EVALUASI BERDASARKAN DATASET USER")

# Filter pengguna yang memiliki minimal 10 rating untuk memastikan pembagian data train/test yang memadai
user_rating_counts = ratings_filtered['User-ID'].value_counts()
active_users = user_rating_counts[user_rating_counts >= 10].index
ratings_active_users = ratings_filtered[ratings_filtered['User-ID'].isin(active_users)]

print(f"Pengguna dengan minimal 10 rating: {len(active_users)}")
print(f"Jumlah rating dari pengguna aktif: {len(ratings_active_users)}")

# Membagi data untuk evaluasi - tanpa stratify agar menghindari error
train_data, test_data = train_test_split(ratings_active_users, test_size=0.2, random_state=42)

print(f"Data pelatihan: {len(train_data)} rating")
print(f"Data pengujian: {len(test_data)} rating")

# Membuat matriks user-item dari data pelatihan
train_matrix = train_data.pivot_table(
    index=['User-ID'],
    columns=['Book-Title'],
    values='Book-Rating',
    fill_value=0
)

print(f"Ukuran matriks pelatihan: {train_matrix.shape}")

# Matriks kemiripan berdasarkan data pelatihan
train_euclidean_sim = calculate_euclidean_similarity(train_matrix)
train_cosine_sim = calculate_cosine_similarity(train_matrix)

# Jika jumlah kolom (buku) banyak, gunakan fast pearson untuk efisiensi
if train_matrix.shape[1] > 200:
    train_corrMatrix = fast_pearson_similarity(train_matrix, top_n=300)
else:
    train_corrMatrix = train_matrix.corr(method='pearson', min_periods=5)

# 13. Fungsi untuk mendapatkan rekomendasi berdasarkan user history
def get_user_recommendations(user_id, similarity_matrix, user_matrix, method_name, n_recommendations=10):
    """
    Menghasilkan rekomendasi untuk seorang user berdasarkan riwayat rating-nya.
    """
    try:
        # Pastikan user ada dalam data
        if user_id not in user_matrix.index:
            return pd.Series(), []

        # Ambil rating user
        user_data = user_matrix.loc[user_id]
        rated = user_data[user_data > 0]
        unrated = user_data[user_data == 0].index

        if rated.empty:
            return pd.Series(), []

        scores = {}

        for book in unrated:
            if book not in similarity_matrix.columns:
                continue

            sim_sum = 0
            weighted_sum = 0

            for rated_book, rating in rated.items():
                if rated_book in similarity_matrix.index:
                    sim = similarity_matrix.at[rated_book, book]
                    if not np.isnan(sim) and sim > 0:
                        weighted_sum += sim * rating
                        sim_sum += abs(sim)

            if sim_sum > 0:
                scores[book] = weighted_sum / sim_sum

        recommendations = pd.Series(scores).sort_values(ascending=False)
        return recommendations.head(n_recommendations), rated.index.tolist()

    except Exception as e:
        print(f"Gagal menghasilkan rekomendasi untuk user {user_id} menggunakan metode {method_name}: {e}")
        return pd.Series(), []

# 14. Evaluasi detail untuk setiap user
print("\n EVALUASI DETAIL UNTUK SETIAP PENGGUNA")

# Pilih pengguna yang memiliki rating di data train dan test, serta jumlah rating yang memadai
test_users = test_data['User-ID'].unique()
train_users = train_data['User-ID'].unique()
evaluation_users = list(set(test_users) & set(train_users))

# Filter pengguna yang memiliki minimal 3 rating di train dan 2 di test
valid_evaluation_users = []
for user_id in evaluation_users:
    train_count = len(train_data[train_data['User-ID'] == user_id])
    test_count = len(test_data[test_data['User-ID'] == user_id])
    if train_count >= 3 and test_count >= 2:
        valid_evaluation_users.append(user_id)

# Batasi evaluasi ke maksimal 20 pengguna
sample_users = valid_evaluation_users[:20]
print(f"Mengevaluasi {len(sample_users)} pengguna yang memiliki cukup rating di data train dan test")
print(f"Kriteria: Minimal 3 rating di train, Minimal 2 rating di test")

# Hasil evaluasi rinci
detailed_results = []
user_evaluation_details = {}

methods = [
    ('Pearson Correlation', train_corrMatrix),
    ('Cosine Similarity', train_cosine_sim),
    ('Euclidean Similarity', train_euclidean_sim)
]

for i, user_id in enumerate(sample_users):
    print(f"\n--- MENGECEK PENGGUNA {user_id} ({i+1}/{len(sample_users)}) ---")

    # Ambil data test pengguna
    user_test_data = test_data[test_data['User-ID'] == user_id]
    actual_books = set(user_test_data['Book-Title'].tolist())
    actual_ratings = dict(zip(user_test_data['Book-Title'], user_test_data['Book-Rating']))

    # Ambil data train pengguna
    user_train_data = train_data[train_data['User-ID'] == user_id]
    train_books = set(user_train_data['Book-Title'].tolist())
    train_ratings = dict(zip(user_train_data['Book-Title'], user_train_data['Book-Rating']))

    print(f"Profil Pengguna {user_id}:")
    print(f"  - Buku dalam data latih: {len(train_books)}")
    print(f"  - Buku dalam data uji: {len(actual_books)}")
    print(f"  - Rata-rata rating (train): {np.mean(list(train_ratings.values())):.2f}")
    print(f"  - Rata-rata rating (test): {np.mean(list(actual_ratings.values())):.2f}")

    user_results = {
        'user_id': user_id,
        'train_books_count': len(train_books),
        'test_books_count': len(actual_books),
        'train_avg_rating': np.mean(list(train_ratings.values())),
        'test_avg_rating': np.mean(list(actual_ratings.values()))
    }

    # Evaluasi tiap metode
    for method_name, similarity_matrix in methods:
        print(f"\n  Hasil menggunakan {method_name}:")

        # Ambil rekomendasi
        recommendations, user_rated_books = get_user_recommendations(
            user_id, similarity_matrix, train_matrix, method_name, n_recommendations=15
        )

        if len(recommendations) == 0:
            print(f"    Tidak ada rekomendasi yang dihasilkan untuk pengguna {user_id}")
            user_results[f'{method_name.lower().replace(" ", "_")}_precision'] = 0
            user_results[f'{method_name.lower().replace(" ", "_")}_recall'] = 0
            user_results[f'{method_name.lower().replace(" ", "_")}_f1'] = 0
            user_results[f'{method_name.lower().replace(" ", "_")}_coverage'] = 0
            user_results[f'{method_name.lower().replace(" ", "_")}_recommendations'] = []
            user_results[f'{method_name.lower().replace(" ", "_")}_scores'] = []
            continue

        # Hitung metrik evaluasi
        recommended_books = set(recommendations.index[:8])  # Top 8 rekomendasi

        relevant_recommended = recommended_books & actual_books
        precision = len(relevant_recommended) / len(recommended_books) if len(recommended_books) > 0 else 0
        recall = len(relevant_recommended) / len(actual_books) if len(actual_books) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        total_books_in_matrix = len([col for col in similarity_matrix.columns if col in train_matrix.columns])
        coverage = len(recommendations) / total_books_in_matrix if total_books_in_matrix > 0 else 0

        print(f"    Buku yang direkomendasikan: {list(recommended_books)[:3]}..." if len(recommended_books) > 3 else f"    Buku yang direkomendasikan: {list(recommended_books)}")
        print(f"    Buku sebenarnya di data uji: {list(actual_books)[:3]}..." if len(actual_books) > 3 else f"    Buku sebenarnya di data uji: {list(actual_books)}")
        print(f"    Buku yang cocok: {list(relevant_recommended)}")
        print(f"    Precision: {precision:.4f} ({len(relevant_recommended)}/{len(recommended_books)})")
        print(f"    Recall: {recall:.4f} ({len(relevant_recommended)}/{len(actual_books)})")
        print(f"    F1-Score: {f1:.4f}")
        print(f"    Coverage: {coverage:.4f}")

        # Simpan hasil evaluasi
        method_key = method_name.lower().replace(" ", "_")
        user_results[f'{method_key}_precision'] = precision
        user_results[f'{method_key}_recall'] = recall
        user_results[f'{method_key}_f1'] = f1
        user_results[f'{method_key}_coverage'] = coverage
        user_results[f'{method_key}_recommendations'] = list(recommendations.head(5).index)
        user_results[f'{method_key}_scores'] = list(recommendations.head(5).values)

    detailed_results.append(user_results)
    user_evaluation_details[user_id] = user_results

# 15. Agregasi hasil evaluasi
print("\nAGGREGATED EVALUATION RESULTS")
eval_df = pd.DataFrame(detailed_results)

# Calculate average metrics for each method
methods_short = ['pearson_correlation', 'cosine_similarity', 'euclidean_similarity']
method_names = ['Pearson Correlation', 'Cosine Similarity', 'Euclidean Similarity']

summary_results = {}

for method_short, method_name in zip(methods_short, method_names):
    precision_col = f'{method_short}_precision'
    recall_col = f'{method_short}_recall'
    f1_col = f'{method_short}_f1'
    coverage_col = f'{method_short}_coverage'

    avg_precision = eval_df[precision_col].mean()
    avg_recall = eval_df[recall_col].mean()
    avg_f1 = eval_df[f1_col].mean()
    avg_coverage = eval_df[coverage_col].mean()

    summary_results[method_name] = {
        'Average Precision': avg_precision,
        'Average Recall': avg_recall,
        'Average F1-Score': avg_f1,
        'Average Coverage': avg_coverage,
        'Std Precision': eval_df[precision_col].std(),
        'Std Recall': eval_df[recall_col].std()
    }

    print(f"\n{method_name}:")
    print(f"  Average Precision: {avg_precision:.4f} (±{eval_df[precision_col].std():.4f})")
    print(f"  Average Recall: {avg_recall:.4f} (±{eval_df[recall_col].std():.4f})")
    print(f"  Average F1-Score: {avg_f1:.4f}")
    print(f"  Average Coverage: {avg_coverage:.4f}")

# 16. Analisis pengguna dengan performa terbaik dan terburuk
print("\n PENGGUNA DENGAN PERFORMA TERBAIK DAN TERBURUK")

# Temukan pengguna dengan performa terbaik untuk setiap metode
for method_short, method_name in zip(methods_short, method_names):
    f1_col = f'{method_short}_f1'

    # Pengguna terbaik
    best_users = eval_df.nlargest(3, f1_col)
    print(f"\n3 Pengguna Teratas untuk {method_name} (berdasarkan F1-Score):")
    for _, user_row in best_users.iterrows():
        print(f"  Pengguna {user_row['user_id']}: F1={user_row[f1_col]:.4f}, "
              f"Presisi={user_row[f'{method_short}_precision']:.4f}, "
              f"Recall={user_row[f'{method_short}_recall']:.4f}")

# 17. Analisis detail pengguna
print("\n=== ANALISIS DETAIL PENGGUNA ===")
print("5 pengguna teratas dengan aktivitas terbanyak:")

# Urutkan pengguna berdasarkan total aktivitas (buku train + test)
eval_df['total_books'] = eval_df['train_books_count'] + eval_df['test_books_count']
top_active_users = eval_df.nlargest(5, 'total_books')

for _, user_row in top_active_users.iterrows():
    user_id = user_row['user_id']
    print(f"\n--- ANALISIS PENGGUNA {user_id} ---")
    print(f"Tingkat Aktivitas: {user_row['total_books']} total buku")
    print(f"Buku dalam Train: {user_row['train_books_count']}")
    print(f"Buku dalam Test: {user_row['test_books_count']}")
    print(f"Rata-rata Rating Train: {user_row['train_avg_rating']:.2f}")
    print(f"Rata-rata Rating Test: {user_row['test_avg_rating']:.2f}")

    print("\nPerforma Metode:")
    for method_short, method_name in zip(methods_short, method_names):
        precision = user_row[f'{method_short}_precision']
        recall = user_row[f'{method_short}_recall']
        f1 = user_row[f'{method_short}_f1']

        print(f"  {method_name}:")
        print(f"    Presisi: {precision:.4f}")
        print(f"    Recall: {recall:.4f}")
        print(f"    F1-Score: {f1:.4f}")

        # Tampilkan rekomendasi teratas
        if f'{method_short}_recommendations' in user_row:
            recs = user_row[f'{method_short}_recommendations']
            scores = user_row[f'{method_short}_scores']
            print(f"    3 Rekomendasi Teratas:")
            for i, (book, score) in enumerate(zip(recs[:3], scores[:3])):
                print(f"      {i+1}. {book[:50]}... (Skor: {score:.4f})")

# 18. Statistik Ringkasan Akhir
print("\n=== STATISTIK RINGKASAN AKHIR ===")
print(f"Statistik Dataset:")
print(f"  Total pengguna yang dievaluasi: {len(sample_users)}")
print(f"  Total buku dalam sistem: {len(popular_books)}")
print(f"  Total rating: {len(ratings_filtered)}")
print(f"  Rata-rata buku per pengguna: {eval_df['total_books'].mean():.2f}")
print(f"  Rata-rata buku train per pengguna: {eval_df['train_books_count'].mean():.2f}")
print(f"  Rata-rata buku test per pengguna: {eval_df['test_books_count'].mean():.2f}")

print(f"\nMetode Terbaik Secara Keseluruhan:")
avg_f1_scores = {
    'Pearson Correlation': eval_df['pearson_correlation_f1'].mean(),
    'Cosine Similarity': eval_df['cosine_similarity_f1'].mean(),
    'Euclidean Similarity': eval_df['euclidean_similarity_f1'].mean()
}

best_method = max(avg_f1_scores, key=avg_f1_scores.get)
print(f"  Metode dengan performa terbaik: {best_method}")
print(f"  Skor F1 terbaik: {avg_f1_scores[best_method]:.4f}")

print(f"\nPeringkat Metode (berdasarkan rata-rata F1-Score):")
sorted_methods = sorted(avg_f1_scores.items(), key=lambda x: x[1], reverse=True)
for i, (method, score) in enumerate(sorted_methods, 1):
    print(f"  {i}. {method}: {score:.4f}")