from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import plotly.express as px

app = Flask(__name__)

# Load model, scaler, dan dataset
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    df_iris_raw = pd.read_csv('Iris.csv')

    # Ambil nilai unik untuk dropdown input
    sepal_length_options = df_iris_raw['SepalLengthCm'].unique().tolist()
    sepal_width_options = df_iris_raw['SepalWidthCm'].unique().tolist()
    petal_length_options = df_iris_raw['PetalLengthCm'].unique().tolist()
    petal_width_options = df_iris_raw['PetalWidthCm'].unique().tolist()

except Exception as e:
    print(f"Error saat memuat file: {e}")
    exit()

# Halaman utama
@app.route('/')
def index():
    return render_template(
        'index.html',
        sepal_length_options=sepal_length_options,
        sepal_width_options=sepal_width_options,
        petal_length_options=petal_length_options,
        petal_width_options=petal_width_options
    )

# Halaman prediksi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data input dari form
        sepal_length = float(request.form['SepalLengthCm'])
        sepal_width = float(request.form['SepalWidthCm'])
        petal_length = float(request.form['PetalLengthCm'])
        petal_width = float(request.form['PetalWidthCm'])

        # Proses input
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        input_scaled = scaler.transform(input_data)

        # Prediksi
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0]
        class_labels = model.classes_
        probabilities = {label: float(p) for label, p in zip(class_labels, prob)}

        # Normalisasi label prediksi agar sesuai dengan HTML
        label_map = {
            'setosa': 'iris-setosa',
            'versicolor': 'iris-versicolor',
            'virginica': 'iris-virginica',
            'Iris-setosa': 'iris-setosa',
            'Iris-versicolor': 'iris-versicolor',
            'Iris-virginica': 'iris-virginica'
        }
        prediction_label = label_map.get(pred.lower(), pred)

        # Visualisasi plot interaktif
        fig = px.scatter(
            df_iris_raw,
            x='SepalLengthCm',
            y='SepalWidthCm',
            color='Species',
            title=f'Hasil Prediksi: {pred}',
            labels={
                'SepalLengthCm': 'Panjang Sepal (cm)',
                'SepalWidthCm': 'Lebar Sepal (cm)'
            },
            template='plotly_white'
        )

        fig.add_scatter(
            x=[sepal_length],
            y=[sepal_width],
            mode='markers',
            marker=dict(color='black', size=15, line=dict(color='white', width=2)),
            name='Input Anda'
        )

        plot_html = fig.to_html(full_html=False)

        return render_template(
            'result.html',
            prediction=prediction_label,
            probabilities=probabilities,
            plot_html=plot_html
        )

    except Exception as e:
        return render_template(
            'index.html',
            error=f"Terjadi kesalahan: {e}",
            sepal_length_options=sepal_length_options,
            sepal_width_options=sepal_width_options,
            petal_length_options=petal_length_options,
            petal_width_options=petal_width_options
        )

# Jalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)
