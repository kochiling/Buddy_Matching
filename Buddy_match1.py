import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import gower
from sklearn.neighbors import NearestNeighbors

# Initialize Firebase Admin SDK
cred = credentials.Certificate("C:\\Users\\choon\\Documents\\Chi Ling\\BCSCUN\\FYP\\buddyin-70-firebase-adminsdk-ghyzj-6fd743f42a.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://buddyin-70-default-rtdb.firebaseio.com/'
})

def fetch_firebase_data():
    ref = db.reference('KNN Data Information')
    data = ref.get()
    return data

def display_and_store_combined_buddies(user_id):
    data = fetch_firebase_data()
    
    if not data:
        print("No data found.")
        return

    current_user_info = data.get(user_id)
    if not current_user_info:
        print(f"User ID {user_id} not found.")
        return

    current_user_course = current_user_info.get('course')
    current_user_seniority = current_user_info.get('seniority')

    if current_user_course is None or current_user_seniority is None:
        print(f"User ID {user_id} does not have course or seniority specified.")
        return

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

    # Add the current user's info to the combined buddies list
    combined_buddies.append({'uid': user_id, **current_user_info})
    print(f"Combined Buddies with Different Seniority compared to {current_user_seniority} in the same course:")
    print(combined_buddies)

    other_users.append({'uid': user_id, **current_user_info})
    print("Other Users:")
    print(other_users)
    

    if combined_buddies:
        df_combined = pd.DataFrame(combined_buddies)
        print("Combined Buddies DataFrame:")
        print(df_combined)
        
        # Calculate Gower similarity for hobbies and personalities within the same course buddies
        df_combined_subset = df_combined[['course','hobbies', 'personalities',"seniority"]]
        df_combined_gower = gower.gower_matrix(df_combined_subset)
        
        # Display similarity matrix with user IDs as row/column labels
        similarity_matrix = pd.DataFrame(df_combined_gower, index=df_combined['uid'], columns=df_combined['uid'])

        # Set pandas to display all rows and columns
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)  # No limit on width of the display
        pd.set_option('display.max_colwidth', None)  # No limit on column width

        # Now, when you print the DataFrame, it should show everything
        print("Gower Similarity Matrix for Combined Buddies (Including Current User):")
        print(similarity_matrix)

    else:
        print(f"No matching buddies found with different seniority compared to {current_user_seniority} in the same course.")


    # If no same-course buddies with different seniority, calculate Gower similarity across all users
    if not other_users:
        print(f"No other users found to calculate Gower similarity across all users.")
    else:
        df_other_users = pd.DataFrame(other_users)
        print("All Users DataFrame:")
        print(df_other_users)
        
        # Calculate Gower similarity for hobbies and personalities across all users
        df_other_users_subset = df_other_users[['course','hobbies', 'personalities',"seniority"]]
        df_other_users_gower = gower.gower_matrix(df_other_users_subset)

        similarity_matrix = pd.DataFrame(df_other_users_gower, index=df_other_users['uid'], columns=df_other_users['uid'])

        # Set pandas to display all rows and columns
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)  # No limit on width of the display
        pd.set_option('display.max_colwidth', None)  # No limit on column width

        # Now, when you print the DataFrame, it should show everything
        print("Gower Similarity Matrix for All Users:")
        print(similarity_matrix)

    # if other_users:
    #     df_all_users = pd.DataFrame(other_users)
    #     print("All Users DataFrame:")
    #     print(df_all_users)

    #     gower.gower_matrix(df_all_users)
    #     print("Gower Similarity Matrix for All Users:")
    #     print(df_all_users)
    #      # Set pandas to display all rows and columns
    #     pd.set_option('display.max_rows', None)
    #     pd.set_option('display.max_columns', None)
    #     pd.set_option('display.width', None)  # No limit on width of the display
    #     pd.set_option('display.max_colwidth', None)  # No limit on column width


if __name__ == "__main__":
    # Specify the user_id you want to exclude from the print
    current_user_id = "X8FHufhudTNt9TXB8jy7OjJpqQH3"
    
    # Call the function to display and store combined buddies with different seniority
    display_and_store_combined_buddies(current_user_id)