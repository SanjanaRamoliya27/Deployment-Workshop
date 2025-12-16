import gradio as gr
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# Load trained models
# -----------------------------
with open("cart_model.pkl", "rb") as f:
    cart_model = pickle.load(f)

with open("id3_model.pkl", "rb") as f:
    id3_model = pickle.load(f)

CLASS_NAMES = ["Setosa", "Versicolor", "Virginica"]

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# -----------------------------
# Helper
# -----------------------------
def get_model(model_type):
    return cart_model if model_type == "CART (Gini)" else id3_model


# -----------------------------
# Single prediction (RETURN STRING, NOT DICT)
# -----------------------------
def predict_single(sl, sw, pl, pw, model_type):
    X = np.array([[sl, sw, pl, pw]])
    model = get_model(model_type)

    probs = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))

    return (
        f"Model: {model_type}\n"
        f"Prediction: {CLASS_NAMES[pred_idx]}\n\n"
        f"Probabilities:\n"
        f"Setosa: {probs[0]:.3f}\n"
        f"Versicolor: {probs[1]:.3f}\n"
        f"Virginica: {probs[2]:.3f}"
    )


# -----------------------------
# Batch prediction
# -----------------------------
def predict_batch(file, model_type):
    if file is None:
        return pd.DataFrame({"error": ["Upload a CSV file"]})

    df = pd.read_csv(file.name)

    # Rename sklearn-style columns if present
    rename_map = {
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
    }
    df = df.rename(columns=rename_map)

    if not all(c in df.columns for c in FEATURES):
        return pd.DataFrame({"error": ["Invalid CSV columns"]})

    X = df[FEATURES].astype(float).to_numpy()
    model = get_model(model_type)

    probs = model.predict_proba(X)
    preds = np.argmax(probs, axis=1)

    df["predicted_class"] = [CLASS_NAMES[i] for i in preds]
    df["prob_setosa"] = probs[:, 0]
    df["prob_versicolor"] = probs[:, 1]
    df["prob_virginica"] = probs[:, 2]

    df.to_csv("batch_predictions.csv", index=False)
    return df


# -----------------------------
# UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🌸 Decision Tree Classifier (CART vs ID3)")

    with gr.Tab("Single prediction"):
        sl = gr.Number(label="Sepal length", value=5.1)
        sw = gr.Number(label="Sepal width", value=3.5)
        pl = gr.Number(label="Petal length", value=1.4)
        pw = gr.Number(label="Petal width", value=0.2)

        model_type = gr.Radio(
            ["CART (Gini)", "ID3 (Entropy)"], value="CART (Gini)"
        )

        btn = gr.Button("Predict")
        out = gr.Textbox(lines=8, label="Result")

        btn.click(
            predict_single,
            inputs=[sl, sw, pl, pw, model_type],
            outputs=out,
        )

    with gr.Tab("Batch prediction"):
        file = gr.File(file_types=[".csv"])
        model_type_b = gr.Radio(
            ["CART (Gini)", "ID3 (Entropy)"], value="CART (Gini)"
        )

        btn_b = gr.Button("Run batch prediction")
        table = gr.Dataframe()
        download = gr.File(label="Download CSV")

        btn_b.click(
            predict_batch,
            inputs=[file, model_type_b],
            outputs=table,
        )

        btn_b.click(
            lambda: "batch_predictions.csv",
            outputs=download,
        )

# ❌ DO NOT call demo.launch() in HF Spaces
