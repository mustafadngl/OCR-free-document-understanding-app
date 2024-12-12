from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
from datasets import load_dataset
import torch
import re
import os

# Initialize Donut model and processor
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Flask app setup
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './upload'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables to hold image processing data
current_image_pixel_values = None
parsed_results = None  # To store parsed document results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    global current_image_pixel_values, parsed_results, dataset
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    dataset = load_dataset("imagefolder", data_dir="./upload", split="train") #load_dataset("hf-internal-testing/example-documents", split="test")
    image = dataset[0]["image"]

    # prepare decoder inputs
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    current_image_pixel_values = processor(image, return_tensors="pt").pixel_values

    outputs = model.generate(
        current_image_pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
    parsed_results = processor.token2json(sequence)
    
    return jsonify({'message': 'Image uploaded and processed successfully!', 'parsed_results': parsed_results}), 200


@app.route('/results', methods=['GET'])
def get_parsed_results():
    if parsed_results is None:
        return jsonify({'error': 'No document parsed yet. Please upload a document first.'}), 400

    return jsonify({'parsed_results': parsed_results})


@app.route('/ask', methods=['POST'])
def ask_question():
    global current_image_pixel_values
    if current_image_pixel_values is None:
        return jsonify({'error': 'No image uploaded yet. Please upload an image first.'}), 400

    data = request.json
    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided.'}), 400

    question = data['question']
    task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
    prompt = task_prompt.replace("{user_input}", question)

    decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids

    outputs = model.generate(
        current_image_pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # Remove first task start token
    response = processor.token2json(sequence)
    return jsonify({'question': question, 'answer': response})


if __name__ == '__main__':
    app.run(debug=True)
