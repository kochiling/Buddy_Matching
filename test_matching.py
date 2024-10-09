import pandas as pd
import gower
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

# Load CSV data prepared from Firebase
def load_csv_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Perform KNN match on buddies using similarity matrix
def knn_match_buddies(similarity_matrix, k, user_id):
    # Exclude the current user's own similarity to themselves (set it to a large value)
    similarity_matrix.loc[user_id, user_id] = 0

    # Get the number of users and ensure k is not greater than the number of users
    num_users = similarity_matrix.shape[0]
    k = max(1, min(k, num_users - 1))

    # Fit KNN model using the distance matrix
    knn = NearestNeighbors(n_neighbors=k, metric='precomputed')
    knn.fit(similarity_matrix)

    # Find k nearest neighbors for each user
    distances, indices = knn.kneighbors(similarity_matrix)

    # Create a dictionary to store the matched buddies for each user
    matched_buddies = {}
    for i, idx in enumerate(similarity_matrix.index):
        if idx == user_id:
            nearest_buddies = similarity_matrix.index[indices[i]].tolist()
            # Remove the user ID from their own nearest buddies list if present
            nearest_buddies = [buddy for buddy in nearest_buddies if buddy != user_id]
            matched_buddies[user_id] = nearest_buddies

    return matched_buddies

# Test accuracy of KNN using cross-validation
def test_knn_accuracy(similarity_matrix, y_labels, k=3):
    knn = KNeighborsClassifier(n_neighbors=k, metric='precomputed')
    
    # Perform 5-fold cross-validation
    X = similarity_matrix.values
    scores = cross_val_score(knn, X, y_labels, cv=5)
    
    print(f"Cross-validation scores: {scores}")
    print(f"Average accuracy: {scores.mean()}")
    # Perform cross-validation and calculate precision, recall, and F1 score
    precision_scores = cross_val_score(knn, X, y_labels, cv=5, scoring='precision')
    recall_scores = cross_val_score(knn, X, y_labels, cv=5, scoring='recall')
    f1_scores = cross_val_score(knn, X, y_labels, cv=5, scoring='f1')

    print(f"Cross-validation precision scores: {precision_scores}")
    print(f"Average precision: {precision_scores.mean()}")
    print(f"Cross-validation recall scores: {recall_scores}")
    print(f"Average recall: {recall_scores.mean()}")
    print(f"Cross-validation F1 scores: {f1_scores}")
    print(f"Average F1 score: {f1_scores.mean()}")

    # Predict the labels using cross-validation
    y_pred = cross_val_score(knn, X, y_labels, cv=5, scoring='accuracy')

    # Calculate the confusion matrix
    cm = confusion_matrix(y_labels, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    return scores.mean(), precision_scores.mean(), recall_scores.mean(), f1_scores.mean()

    # return scores.mean()
    
# Load the data, perform Gower similarity calculation, and evaluate accuracy
def evaluate_matching_algorithm(csv_file, user_id):
    # Step 1: Load CSV Data
    df = load_csv_data(csv_file)

    # Step 2: Label encode categorical columns
    label_encoders = {}
    for column in ['course', 'hobbies', 'personalities']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Step 3: Convert numerical columns to appropriate types
    df['seniority'] = df['seniority'].astype(float)
    df[['course', 'hobbies', 'personalities']] = df[['course', 'hobbies', 'personalities']].astype(float)

    # Step 4: Calculate Gower similarity matrix
    df_subset = df[['course', 'hobbies', 'personalities', 'seniority']]
    gower_matrix = gower.gower_matrix(df_subset)

    # Step 5: Prepare similarity matrix with user IDs as index
    similarity_matrix = pd.DataFrame(gower_matrix, index=df['uid'], columns=df['uid'])
    print("Similarity Matrix:")
    print(similarity_matrix)

    # Step 6: Evaluate KNN model using cross-validation
    # Let's assume y_labels is available as part of the CSV file
    # These labels are the ground truth for "matched" status between users (1 for match, 0 for non-match)
    # For simplicity, we create artificial labels (as ground truth may not exist yet)
    y_labels = [1 if uid != user_id else 0 for uid in df['uid']]  # Replace this with real labels if available
    
    accuracy = test_knn_accuracy(similarity_matrix, y_labels, k=3)
    print(f"Matching Algorithm Accuracy: {accuracy}")

    # Step 7: Use KNN to match buddies for a specific user
    matched_buddies = knn_match_buddies(similarity_matrix, k=3, user_id=user_id)
    print(f"Matched buddies for user {user_id}: {matched_buddies}")

# Example usage: testing the matching algorithm
if __name__ == "__main__":
    # Path to CSV file extracted from Firebase
    csv_file_path = "KNN_test_accuracy.csv"

    # User ID to test the matching for (replace with actual user ID from your data)
    test_user_id = 'jpdef4kROfV7zpQsfDuhgImBqZu1'
    
    # Run the evaluation
    evaluate_matching_algorithm(csv_file_path, test_user_id)