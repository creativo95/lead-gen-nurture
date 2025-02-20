import pandas as pd
import sqlite3
import smtplib
from email.mime.text import MIMEText
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and preprocess data
def load_data():
    df = pd.read_csv('leads_data.csv')
    df.fillna('', inplace=True)
    return df

def preprocess_data(df):
    features = ['website_visits', 'email_opens', 'click_rate']
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

def train_model(df):
    X = df[['website_visits', 'email_opens', 'click_rate']]
    y = df['conversion']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')
    
    return model

def store_leads(df):
    conn = sqlite3.connect('leads.db')
    df.to_sql('leads', conn, if_exists='replace', index=False)
    conn.close()

def send_email(recipient, subject, body):
    sender_email = 'your_email@example.com'
    password = 'your_password'
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient
    
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, recipient, msg.as_string())

if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)
    model = train_model(df)
    store_leads(df)
    print('Lead Generation System is ready!')
