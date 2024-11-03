from flask import Flask, jsonify, request
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import gower
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

@app.route('/')
def home():
    return "Buddy Match API is running!"

# Initialize Firebase Admin SDK
cred = credentials.Certificate("C:\\Users\\choon\\Documents\\Chi Ling\\BCSCUN\\FYP\\buddyin-70-firebase-adminsdk-ghyzj-6fd743f42a.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://buddyin-70-default-rtdb.firebaseio.com/'
})

# Fetch data from Firebase
def fetch_firebase_data():
    ref = db.reference('KNN Data Information')
    data = ref.get()
    return data

def knn_match_buddies(similarity_matrix, k, user_id):
    # Exclude the current user's own similarity to themselves (set it to a large value)
    similarity_matrix.loc[user_id, user_id] = 0

    # Get the number of users and ensure k is not greater than the number of users
    num_users = similarity_matrix.shape[0]
    k = max(1, min(k, num_users - 1))

    # Calculate the distance matrix (similarity) and replace NaN values with a default value
    distance_matrix = similarity_matrix

    # Fit KNN model using the distance matrix
    knn = NearestNeighbors(n_neighbors=k, metric='precomputed')
    knn.fit(distance_matrix)

    # Find k nearest neighbors for each user
    distances, indices = knn.kneighbors(distance_matrix)

     # Debugging prints
    print(f"Distances: {distances}")
    print(f"Indices: {indices}")

    # Create a dictionary to store the matched buddies for each user
    matched_buddies = {}
    for i, idx in enumerate(similarity_matrix.index):
        if idx == user_id:
            nearest_buddies = similarity_matrix.index[indices[i]].tolist()
            # Remove the user ID from their own nearest buddies list if present
            nearest_buddies = [buddy for buddy in nearest_buddies if buddy != user_id]
            matched_buddies[user_id] = nearest_buddies

    print (matched_buddies)

    return matched_buddies

def store_knn_matches_in_firebase(user_id, matched_buddies):
    # Reference to the Firebase path where you want to store the data
    ref = db.reference(f'Buddies/MatchResults/{user_id}')

    # Get the existing matches for the current user ID
    existing_matches = ref.get()

    for buddy_id in matched_buddies[user_id]:
        # Check if the buddy_id is already stored; if not, add it
        if existing_matches is None or buddy_id not in existing_matches:
            ref.update({
                buddy_id: True
            })
    #return jsonify("Matches stored successfully!")

@app.route('/get_buddies', methods=['POST'])
def get_buddies():

    user_id = request.json.get('user_id')
    if user_id is None:
        return jsonify("User ID is required!"), 400

    data = fetch_firebase_data()
    if data is None:
        return jsonify("No data found!"), 400
    
    # Retrieve the current user's info from the data
    current_user_info = data.get(user_id)
    if not current_user_info:
        return jsonify(f"User ID {user_id} not found in the data."), 404
    
    current_user_course = current_user_info.get('course')
    current_user_seniority = current_user_info.get('seniority')

    if current_user_course is None or current_user_seniority is None:
        return jsonify(f"User ID {user_id} does not have course or seniority specified."), 400
    
    combined_buddies = []
    other_users = []

    for uid, user_info in data.items():
        if uid != user_id:
            user_seniority = user_info.get('seniority')
            if user_info.get('course') == current_user_course and user_seniority is not None:
                if user_seniority > current_user_seniority or user_seniority < current_user_seniority:
                    combined_buddies.append({'uid': uid, **user_info})
                else:
                    other_users.append({'uid': uid, **user_info})
            else:
                other_users.append({'uid': uid, **user_info})
    
    combined_buddies.append({'uid': user_id, **current_user_info})
    other_users.append({'uid': user_id, **current_user_info})

    if combined_buddies:
        df_combined = pd.DataFrame(combined_buddies)
         # Label encode categorical variables for Gower similarity calculation
        label_encoders = {}
        for column in ['course', 'hobbies', 'personalities']:
            le = LabelEncoder()
            df_combined[column] = le.fit_transform(df_combined[column])
            label_encoders[column] = le

        # Ensure all numerical columns are of type float64
        df_combined['seniority'] = df_combined['seniority'].astype(float)
        df_combined[['course', 'hobbies', 'personalities']] = df_combined[['course', 'hobbies', 'personalities']].astype(float)

        # Calculate Gower similarity for hobbies and personalities within the same course buddies
        df_combined_subset = df_combined[['course', 'hobbies', 'personalities', 'seniority']]
        df_combined_gower = gower.gower_matrix(df_combined_subset)

        # Display similarity matrix with user IDs as row/column labels
        similarity_matrix = pd.DataFrame(df_combined_gower, index=df_combined['uid'], columns=df_combined['uid'])
        print("Similarity Matrix:")
        print(similarity_matrix)

        # Implement KNN to find matched buddies
        k = 4 # Call a function to get the value of k
        matched_buddies = knn_match_buddies(similarity_matrix, k, user_id)
        store_knn_matches_in_firebase(user_id, matched_buddies)
        
    else:
        return jsonify("No combined buddies found!"), 400
    
    if other_users:
        df_other_users = pd.DataFrame(other_users)
        
        # Label encode categorical variables for Gower similarity calculation
        for column in ['course', 'hobbies', 'personalities']:
            le = LabelEncoder()
            df_other_users[column] = le.fit_transform(df_other_users[column])

        # Ensure all numerical columns are of type float64
        df_other_users['seniority'] = df_other_users['seniority'].astype(float)

        # Calculate Gower similarity for hobbies and personalities across all users
        df_other_users_subset = df_other_users[['course', 'hobbies', 'personalities', 'seniority']]
        df_other_users_gower = gower.gower_matrix(df_other_users_subset)

        similarity_matrix = pd.DataFrame(df_other_users_gower, index=df_other_users['uid'], columns=df_other_users['uid'])
        print("Similarity Matrix:")
        print(similarity_matrix)
        k = 4
        matched_buddies = knn_match_buddies(similarity_matrix, k, user_id)
        store_knn_matches_in_firebase(user_id, matched_buddies)
        return jsonify("Match Successfully!")

    else:
        return jsonify("No other users found!"), 400

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)