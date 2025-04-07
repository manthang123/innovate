from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask_cors import CORS
import re

app = Flask(__name__)
CORS(app)

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("fine_tuned_gpt2_qa")

# Function to generate an answer given a question
def generate_answer(question, max_length=50):
    input_text = f"Question: {question} Answer:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = answer.split("Answer:")[1].strip()
    return answer

# Function to extract link from answer
def extract_link(answer):
    match = re.search(r'(https?://\S+)', answer)
    return match.group(0) if match else None

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Route to process input
@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        user_input = request.form['user_input']
        predicted_answer = generate_answer(user_input)
        link = extract_link(predicted_answer)
        return render_template('index.html', user_input=user_input, predicted_answer=predicted_answer, link=link)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
