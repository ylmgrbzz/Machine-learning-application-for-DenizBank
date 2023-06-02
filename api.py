from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json
app = Flask(__name__)

# Bağımlı değişken
target = 'default'

# Bağımsız değişkenler
categorical_columns = ['status_last_archived_0_24m',
                       'status_2nd_last_archived_0_24m',
                       'status_3rd_last_archived_0_24m',
                       'status_max_archived_0_6_months',
                       'status_max_archived_0_12_months',
                       'status_max_archived_0_24_months',
                       'name_in_email',
                       'worst_status_active_inv',
                       'merchant_group', 'merchant_category',
                       'account_worst_status_6_12m',
                       'account_worst_status_3_6m',
                       'account_worst_status_12_24m',
                       'account_worst_status_0_3m',
                       'account_status']

# Kaydedilmiş sınıflandırıcıyı yükleyin
with open('classifier.pkl', 'rb') as file:
    classifier = pickle.load(file)

with open('encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)


@app.route('/predict', methods=['POST'])
def predict():
    # Gelen JSON verisini alın
    data = request.get_json()

    # Özellikleri çıkarın
    #print keys

    print(data.keys())
    features = data['data']
    df = pd.DataFrame(features)
    df = df.drop('uuid', axis=1)
    df = df.drop('default', axis=1)

    # Kategorik özellikleri kodlayın
    df_encoded = pd.DataFrame(encoder.transform(df[categorical_columns]))

    # Sınıflandırma yapın
    prediction = classifier.predict(df_encoded)

    # Tahmini JSON formatında dönün
    response = {'prediction': prediction.tolist()}

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
