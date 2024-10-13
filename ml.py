import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the financial transactions dataset (replace with your actual dataset)
data = {
    'transaction_amount': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'transaction_type': ['online', 'offline', 'online', 'offline', 'online', 'offline', 'online', 'offline', 'online', 'offline'],
    'user_behavior': [0.1, 0.5, 0.2, 0.8, 0.3, 0.4, 0.2, 0.9, 0.1, 0.6],
    'geographical_location': ['urban', 'rural', 'urban', 'rural', 'urban', 'rural', 'urban', 'rural', 'urban', 'rural'],
    'is_fraud': [0, 0, 1, 0, 0, 1, 0, 1, 0, 0]  # 1 indicates fraud
}

# Create a DataFrame
fraud_df = pd.DataFrame(data)

# One-hot encode categorical variables
fraud_df = pd.get_dummies(fraud_df, columns=['transaction_type', 'geographical_location'], drop_first=True)

# Split the dataset into features and target
X = fraud_df.drop("is_fraud", axis=1)
y = fraud_df["is_fraud"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Streamlit app configuration
PAGE_CONFIG = {
    "page_title": "Zero Financial Fraud Detection",
    "layout": "centered",
    "initial_sidebar_state": "auto"
}
st.set_page_config(**PAGE_CONFIG)

# Add custom CSS for the background with a design theme
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://www.toptal.com/designers/subtlepatterns/uploads/double-bubble.png");
    background-size: cover;
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""
st.markdown(background_image, unsafe_allow_html=True)

def main():
    st.title("Zero Financial Fraud Detection")
    st.write("Enter the following transaction details to predict the risk of fraud:")

    # Input fields for user to enter transaction information
    transaction_amount = st.number_input("*Transaction Amount*", min_value=0.0, value=0.0, step=0.1)
    transaction_type = st.selectbox("*Transaction Type*", ["online", "offline"])
    user_behavior = st.number_input("*User Behavior Score (0 to 1)*", min_value=0.0, max_value=1.0, step=0.01)
    geographical_location = st.selectbox("*Geographical Location*", ["urban", "rural"])

    # Create a button to trigger the prediction
    if st.button("*Predict*"):
        # Prepare the input data for prediction
        user_data = pd.DataFrame(
            {
                "transaction_amount": [transaction_amount],
                "user_behavior": [user_behavior],
                "transaction_type_online": [1 if transaction_type == 'online' else 0],
                "transaction_type_offline": [1 if transaction_type == 'offline' else 0],
                "geographical_location_urban": [1 if geographical_location == 'urban' else 0],
                "geographical_location_rural": [1 if geographical_location == 'rural' else 0],
            }
        )

        # Ensure all columns from training data are present in user_data
        for col in X.columns:
            if col not in user_data.columns:
                user_data[col] = 0
        
        # Make sure columns are in the same order as X
        user_data = user_data[X.columns]

        # Make the prediction
        prediction = clf.predict(user_data)

        # Display the prediction result
        if prediction[0] == 0:
            st.success("*This transaction is NOT fraudulent.*")
            st.markdown(
            """
            <div style="background-color: #ffecf2; padding: 10px; border-radius: 5px;">
                <h4>Transaction Advice:</h4>
                <ul>
                <li>Continue monitoring transactions for unusual activity.</li>
                <li>Ensure secure practices for online transactions.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True
            )
        else:
            st.error("*This transaction is fraudulent!*")
            st.markdown(
            """
            <div style="background-color: #ffecf2; padding: 10px; border-radius: 5px;">
                <h4>Immediate Actions:</h4>
                <ul>
                <li>Contact your financial institution immediately.</li>
                <li>Freeze your account if necessary.</li>
                <li>Change your passwords and monitor other accounts for suspicious activity.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()
