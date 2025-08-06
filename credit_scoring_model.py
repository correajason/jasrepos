import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        data.columns = data.columns.str.strip()  # Clean column names
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    df.columns = df.columns.str.strip()  # Clean column names
    df.dropna(inplace=True)  # Drop rows with missing values

    # Clean and map loan_status column
    df['loan_status'] = df['loan_status'].astype(str).str.strip().str.capitalize()
    df = df[df['loan_status'].isin(['Approved', 'Rejected'])]  # Remove bad labels
    df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})

    # Encode categorical variables
    categorical_cols = ['education', 'self_employed']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Separate features and target
    X = df_encoded.drop(['loan_id', 'loan_status'], axis=1)
    y = df_encoded['loan_status']

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y


def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'credit_scoring_model.pkl')

    y_pred = model.predict(X_test)
    print("\nModel Evaluation Report:")
    print(classification_report(y_test, y_pred))

    return model



def main():
    filepath = 'loan_approval_dataset.csv' 
    data = load_data(filepath)

    if data is not None:
        print("Sample data:")
        print(data.head())

        X, y = preprocess_data(data)
        model = train_and_evaluate_model(X, y)

if __name__ == '__main__':
    main()

