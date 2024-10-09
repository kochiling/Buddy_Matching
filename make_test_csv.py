import firebase_admin
from firebase_admin import credentials, db
import csv

# Initialize Firebase Admin SDK
cred = credentials.Certificate("C:\\Users\\choon\\Documents\\Chi Ling\\BCSCUN\\FYP\\buddyin-70-firebase-adminsdk-ghyzj-6fd743f42a.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://buddyin-70-default-rtdb.firebaseio.com/'
})

# Fetch data from Firebase and save it as CSV
def export_to_csv():
    ref = db.reference('KNN Data Information')
    data = ref.get()

    with open('KNN_test_accuracy.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['uid', 'course', 'hobbies', 'personalities', 'seniority','name'])  # Adjust based on your data structure

        for uid, user_info in data.items():
            writer.writerow([
                uid,
                user_info.get('course', ''),
                user_info.get('hobbies', ''),
                user_info.get('personalities', ''),
                user_info.get('seniority', ''),
                user_info.get('name', '')
            ])

    print("KNN_test_accuracy.csv")
    print("Total users:", len(data))

# Call the function
export_to_csv()