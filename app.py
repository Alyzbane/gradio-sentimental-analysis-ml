import gradio as gr
import joblib
import os
import re
import string

# Load the models and vectorizers
def load_model_and_vectorizer(path, model_filename='model.pkl', vectorizer_filename='vectorizer.pkl'):
    model = joblib.load(os.path.join(path, model_filename))
    vectorizer = joblib.load(os.path.join(path, vectorizer_filename))
    return model, vectorizer

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

# Load all models at startup
models = {
    "Linear Regression": load_model_and_vectorizer(path=os.path.join('models', 'lr')),
    "MultinomialNB": load_model_and_vectorizer(path=os.path.join('models', 'mnb')),
    "SVM": load_model_and_vectorizer(path=os.path.join('models', 'svm')),
    "Random Forest": load_model_and_vectorizer(path=os.path.join('models', 'rf'))
}

def predict_sentiment(message, model_name="MultinomialNB"):
    model, vectorizer = models[model_name]
    preprocessed = preprocess_text(message)
    vectorized = vectorizer.transform([preprocessed])
    prediction = model.predict(vectorized)[0]
    return prediction

def get_bot_response(message, chat_history, model_choice):
    message = message["text"]
    if not message.strip():
        bot_response = "😺 Please share a game review!"
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_response})
        return "", chat_history
    
    # Get sentiment prediction
    sentiment = predict_sentiment(message, model_choice)
    
    # Generate response based on sentiment
    if sentiment == 1:
        bot_response = f"😸 This is a Positive review!"
    else:
        bot_response = f"😾 This is a Negative review!"
   
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": bot_response})
    return "", chat_history

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Default(), title="Gaming Sentiment Chatbot", css=".upload-button {display: none;} .centered-md {text-align: center}") as demo:
    gr.Markdown("# 🎮 Steam Review Sentiment Analysis", elem_classes="centered-md")
    gr.Markdown("""
    <div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
    ✨ Enter a Steam review to analyze its sentiment. For more information, see the dataset used at:
        <a href="https://www.kaggle.com/datasets/filipkin/steam-reviews" target="_blank">
            <img src="https://img.shields.io/badge/Kaggle-Steam%20Reviews-blue?logo=kaggle" alt="Kaggle">
        </a>
        |
        <a href="https://github.com/alyzbane/gradio-sentimental-analysis-ml" target="_blank">
            <img src="https://img.shields.io/badge/GitHub-Repository-blue?logo=github" alt="GitHub">
        </a>
    </div>
    """, elem_classes="centered-md")

   
    chatbot = gr.Chatbot(
        type="messages",
        label="History",
        placeholder="Share a though about video game 🎮👇",
        height=400,
    )
    
    with gr.Row():
        message = gr.MultimodalTextbox(
        interactive=True,
        placeholder="Enter message...",
        show_label=False,
        )

    with gr.Row():
        model_choice = gr.Dropdown(
            choices=list(models.keys()),
            value="MultinomialNB",
            label=r"↓ Select Model for Analysis",
        )
    
    # Example messages
    gr.Markdown("## Example Messages")
    examples = gr.Examples(
        examples=[
            "This game is absolutely fantastic! The graphics and gameplay are incredible!",
            "I can't believe how buggy this game is. Constant crashes and poor optimization.",
            "Decent game but nothing special. Might be worth it on sale.",
            "Best game I've played this year! The story is amazing!",
            "this game is 1/10 at best. Waste of money"
        ],
        inputs=message,
        label="Example Messages"
    )
   
    # Also allow Enter key to submit
    message.submit(
        fn=get_bot_response,
        inputs=[message, chatbot, model_choice],
        outputs=[message, chatbot]
    )

if __name__ == "__main__":
    demo.launch(debug=False)
