import os
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)
model = load_model("my_model2.h5", compile=False)
scaler = joblib.load('scaler.joblib')
# Labels
labels = {
    0: 'Administrasi Bisnis',
    1: 'Akuntansi',
    2: 'Antropologi',
    3: 'Arsitektur',
    4: 'Bimbingan Konseling',
    5: 'Biologi',
    6: 'Desain Grafis',
    7: 'Digital Bisnis',
    8: 'Ekonomi',
    9: 'Farmasi',
    10: 'Filsafat',
    11: 'Fisika',
    12: 'Geografi',
    13: 'Hubungan Internasional',
    14: 'Hukum',
    15: 'Ilmu Kesejahteraan Sosial',
    16: 'Ilmu Komunikasi',
    17: 'Ilmu Politik',
    18: 'Jurnalistik',
    19: 'Kedokteran',
    20: 'Keperawatan',
    21: 'Kesehatan Masyarakat',
    22: 'Kimia',
    23: 'Kriminologi',
    24: 'Linguistik',
    25: 'Manajemen',
    26: 'Manajemen Bisnis',
    27: 'Hubungan Masyarakat',
    28: 'Marketing',
    29: 'Matematika',
    30: 'Musik',
    31: 'Pariwisata',
    32: 'Pendidikan Olahraga',
    33: 'Pendidikan Anak Usia Dini',
    34: 'Psikologi',
    35: 'Sastra Inggris',
    36: 'Sejarah',
    37: 'Seni Rupa',
    38: 'Sosiologi',
    39: 'Teknik Biomedik',
    40: 'Teknik Elektro',
    41: 'Teknik Elektronika',
    42: 'Teknik Industri',
    43: 'Teknik Informatika',
    44: 'Teknik Kimia',
    45: 'Teknik Komputer',
    46: 'Teknik Lingkungan',
    47: 'Teknik Mesin',
    48: 'Teknik Sipil',
    49: 'Teknologi Informasi'
}


R_features = ['R1', 'R2', 'R4', 'R6', 'R7', 'R8']
I_features = ['I1', 'I2', 'I4', 'I5', 'I7', 'I8']
A_features = ['A2', 'A3', 'A4', 'A5', 'A6', 'A8']
S_features = ['S1', 'S3', 'S5', 'S6', 'S7', 'S8']
E_features = ['E1', 'E3', 'E4', 'E5', 'E7', 'E8']
C_features = ['C2', 'C3', 'C5','C6', 'C7', 'C8']
TIPI_features = ['TIPI1', 'TIPI2', 'TIPI3', 'TIPI4', 'TIPI5', 'TIPI6', 'TIPI7', 'TIPI8', 'TIPI9', 'TIPI10']
VCL_features = ['VCL1', 'VCL2', 'VCL3', 'VCL4', 'VCL5', 'VCL6','VCL10', 'VCL11', 'VCL12', 'VCL13', 'VCL14', 'VCL15']



@app.route("/")
def index():
    return jsonify({
        "status": {
            "code": 201,
            "message": "API is running"
        },
        "data": None
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            data = request.json
            # Ensure all required features are in the input data
            required_features = R_features + I_features + A_features + \
                S_features + E_features + C_features + TIPI_features + VCL_features

            for feature in required_features:
                if feature not in data:
                    return jsonify({
                        "status": {
                            "code": 400,
                            "message": f"Feature '{feature}' is missing from input data"
                        },
                        "data": None
                    }), 400

            # Convert data to DataFrame for easier manipulation
            df = pd.DataFrame([data])

            # Sum the feature groups
            df['R'] = df[R_features].sum(axis=1)/6
            df['I'] = df[I_features].sum(axis=1)/6
            df['A'] = df[A_features].sum(axis=1)/6
            df['S'] = df[S_features].sum(axis=1)/6
            df['E'] = df[E_features].sum(axis=1)/6
            df['C'] = df[C_features].sum(axis=1)/6
            df['TIPI'] = df[TIPI_features].sum(axis=1)
            df['VCL'] = df[VCL_features].sum(axis=1)

            # Select the final 37 features
            final_features = ['R', 'I', 'A', 'S', 'E', 'C', 'VCL', 'TIPI']
            input_data = df[final_features].values
            input_data_scaled = scaler.transform(input_data)
            print(input_data)

            print(input_data_scaled)
            # Predict using the model
            prediction = model.predict(input_data_scaled)
            print("Prediction probabilities:", prediction)

            # Get the index of the maximum value along axis 1
            y_pred = np.argmax(prediction, axis=1)
            print("Predicted indices:", y_pred)

            # Convert indices to class labels
            y_pred_labels = [labels[i] for i in y_pred]
            print("Predicted labels:", y_pred_labels)

            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Prediction successful"
                },
                "data": {
                    "prediction": y_pred_labels
                }
            })
        except Exception as e:
            return jsonify({
                "status": {
                    "code": 500,
                    "message": str(e)
                },
                "data": None
            }), 500

    return jsonify({
        "status": {
            "code": 405,
            "message": "Method not allowed"
        },
        "data": None
    }), 405

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
    # app.run(debug=True)
