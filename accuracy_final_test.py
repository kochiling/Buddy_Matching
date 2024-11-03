import pandas as pd
import gower
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

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
    _, indices = knn.kneighbors(similarity_matrix)

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
    print("Initializing KNN Classifier...")
    knn = KNeighborsClassifier(n_neighbors=k, metric='precomputed')
    
    # Perform 5-fold cross-validation
    X = similarity_matrix.values
    print("Performing 5-fold cross-validation...")
    scores = cross_val_score(knn, X, y_labels, cv=5)
    accuracy = scores.mean()
    
    print(f"Cross-validation scores: {scores}")
    print(f"Average accuracy: {accuracy}")
    
    # Calculate precision, recall, and F1 score
    precision_scores = cross_val_score(knn, X, y_labels, cv=5, scoring='precision')
    recall_scores = cross_val_score(knn, X, y_labels, cv=5, scoring='recall')
    f1_scores = cross_val_score(knn, X, y_labels, cv=5, scoring='f1')

    precision = precision_scores.mean()
    recall = recall_scores.mean()
    f1 = f1_scores.mean()

    print(f"Cross-validation precision: {precision}")
    print(f"Cross-validation recall: {recall}")
    print(f"Cross-validation F1 score: {f1}")

    # Combine all metrics into a single average score
    combined_score = (accuracy + precision + recall + f1) / 4
    print(f"Combined score (Accuracy, Precision, Recall, F1): {combined_score}")

    # Fit the model and predict the labels
    knn.fit(X, y_labels)
    y_pred = knn.predict(X)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_labels, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    print(f"True Positives: {cm[1, 1]}")
    print(f"True Negatives: {cm[0, 0]}")
    print(f"False Positives: {cm[0, 1]}")
    print(f"False Negatives: {cm[1, 0]}")

    # Return the combined score
    return combined_score

# Load the data, perform Gower similarity calculation, and evaluate accuracy
def evaluate_matching_algorithm(csv_file, user_id):
    print("Loading CSV Data...")
    df = load_csv_data(csv_file)
    print("Data loaded, first 5 rows:")
    print(df.head())  # Check data loaded correctly

    # Step 2: Label encode categorical columns
    print("Label encoding categorical columns...")
    label_encoders = {}
    for column in ['course', 'hobbies', 'personalities']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Step 3: Convert numerical columns to appropriate types
    print("Converting columns to float...")
    df['seniority'] = df['seniority'].astype(float)
    df[['course', 'hobbies', 'personalities']] = df[['course', 'hobbies', 'personalities']].astype(float)

    # Step 4: Calculate Gower similarity matrix
    print("Calculating Gower similarity matrix...")
    df_subset = df[['course', 'hobbies', 'personalities', 'seniority']]
    gower_matrix = gower.gower_matrix(df_subset)
    print("Gower matrix calculated, shape:", gower_matrix.shape)

    # Step 5: Prepare similarity matrix with user IDs as index
    print("Preparing similarity matrix...")
    similarity_matrix = pd.DataFrame(gower_matrix, index=df['uid'], columns=df['uid'])
    print("Similarity matrix ready, shape:", similarity_matrix.shape)

    # Step 6: Evaluate KNN model using cross-validation
    print("Evaluating KNN model using cross-validation...")
    y_labels = []
    for uid in df['uid']:
        if df.loc[df['uid'] == uid, 'course'].values[0] == df.loc[df['uid'] == user_id, 'course'].values[0]:
            y_labels.append(1)
        elif df.loc[df['uid'] == uid, 'hobbies'].values[0] == df.loc[df['uid'] == user_id, 'hobbies'].values[0]:
            y_labels.append(1)
        elif df.loc[df['uid'] == uid, 'personalities'].values[0] == df.loc[df['uid'] == user_id, 'personalities'].values[0]:
            y_labels.append(1)
        elif df.loc[df['uid'] == uid, 'seniority'].values[0] == df.loc[df['uid'] == user_id, 'seniority'].values[0]:
            y_labels.append(1)
        else:
            y_labels.append(0)
    
    combined_score = test_knn_accuracy(similarity_matrix, y_labels, k=3)
    print(f"Combined Matching Algorithm Score: {combined_score}")

    # Step 7: Use KNN to match buddies for a specific user
    print(f"Matching buddies for user {user_id}...")
    matched_buddies = knn_match_buddies(similarity_matrix, k=3, user_id=user_id)
    print(f"Matched buddies for user {user_id}: {matched_buddies}\n")

# Example usage: testing the matching algorithm
if __name__ == "__main__":
    # Path to CSV file extracted from Firebase
    csv_file_path = "KNN_test_accuracy.csv"

    # User ID to test the matching for 
    test_user_id = 'SlQxfLITEcOzGKPxSq9Y7APXPQo1'
    
    # Run the evaluation
    evaluate_matching_algorithm(csv_file_path, test_user_id)
