import gradio as gr
import joblib
import os
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the models and vectorizers
def load_model_and_vectorizer(path, model_filename='model.pkl', vectorizer_filename='vectorizer.pkl'):
    model = joblib.load(os.path.join(path, model_filename))
    vectorizer = joblib.load(os.path.join(path, vectorizer_filename))
    return model, vectorizer

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

# Prediction pipeline
def pipeline(model, inputs, vectorizer):
    results = {}
    for text in inputs:
        preprocessed = preprocess_text(text)
        vectorized = vectorizer.transform([preprocessed])
        prediction = model.predict(vectorized)
        confidence_scores = model.predict_proba(vectorized)[0]  # Get confidence scores for both classes
        results[text] = {
            "Positive Sentiment": confidence_scores[1],
            "Negative Sentiment": confidence_scores[0]
        }
    return results

# Load the models and vectorizers
model1, vectorizer1 = load_model_and_vectorizer(path=os.path.join('models', 'lr'))
model2, vectorizer2 = load_model_and_vectorizer(path=os.path.join('models', 'mnb'))

# Create a Gradio interface
def analyze_sentiment(text, model_choice):
    if model_choice == "Linear Regression":
        return pipeline(model1, [text], vectorizer1)[text]
    elif model_choice == "MultinomialNB":
        return pipeline(model2, [text], vectorizer2)[text]

with gr.Blocks(theme=gr.themes.Glass(), title="Sentimental Analysis") as demo:
    gr.Markdown("# 🎮 Sentiment Analysis for Steam Reviews")
    gr.Markdown("✨ Enter a Steam review to analyze its sentiment. For more information, see the dataset used at [Kaggle Steam Reviews](https://www.kaggle.com/datasets/filipkin/steam-reviews)")
    
    gr.Markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/your-username/your-repository)")

    with gr.Row():
        with gr.Column():
            review_input = gr.Textbox(lines=2, placeholder="Enter Steam review here...", label="💬 Steam Review Input")
            model_choice = gr.Dropdown(choices=["Linear Regression", "MultinomialNB"], label="Select Model")
            analyze_button = gr.Button("Analyze")
        with gr.Column():
            sentiment_output = gr.Label(label="📊 Sentiment Analysis Result")
    
    analyze_button.click(
        analyze_sentiment, 
        inputs=[review_input, model_choice], 
        outputs=sentiment_output
    )
    
    gr.Markdown("## Example Reviews")
    example_table = gr.Examples(
        examples=[
            ["This game is amazing! I loved every moment of it."],
            ["The graphics are decent, but the gameplay is very repetitive."],
            ["Terrible game. It crashes all the time and is unplayable."]
        ],
        inputs=review_input,
        label="Example Reviews",
    )

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()
