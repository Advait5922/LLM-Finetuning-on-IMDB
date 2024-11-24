# LLM Finetuning for Sentiment Analysis

## Overview
This project implements a sentiment analysis model using a pre-trained large language model (LLM) and fine-tunes it using **LoRA** (Low-Rank Adaptation) for efficient domain adaptation. The model is trained on a IMDB dataset from huggingface, and predictions are made using the fine-tuned model. The approach also integrates the use of **PEFT (Parameter-Efficient Fine-Tuning)** to reduce the computational cost of training large models.

## Technologies Used
- **Hugging Face Transformers**: For loading pre-trained models and tokenizers.
- **PEFT (Low-Rank Adaptation - LoRA)**: For efficient model fine-tuning.
- **PyTorch**: For model training and evaluation.
- **Datasets**: To load and preprocess the dataset.
- **Evaluate**: To calculate the accuracy metric.
- **Hugging Face Hub**: To upload the trained model.

## Requirements
To run the code, ensure the following libraries are installed:
```bash
pip install transformers datasets peft torch evaluate huggingface_hub
```

## Environment Variables
1. **Dataset**: Path of the dataset in Hugging Face Datasets format.
2. **LLM_Model**: Pre-trained model checkpoint name (here `bert-base-uncased`).

## Project Structure
- **Dataset Loading**: Loads the dataset specified by the `Dataset` environment variable.
- **Model Initialization**: Loads a pre-trained model and tokenizer using the `LLM_Model` environment variable.
- **Data Preprocessing**: Tokenizes the dataset and prepares it for training.
- **Model Fine-tuning**: Applies LoRA for parameter-efficient fine-tuning.
- **Training and Evaluation**: Trains the model on the dataset, evaluates performance, and outputs accuracy.
- **Model Push to Hub**: After training, the model is pushed to Hugging Face Hub for sharing.

## Usage

1. **Set Up Environment Variables**: Before running the script, make sure to set up the necessary environment variables:

   ```bash
   export Dataset="path/to/dataset"
   export LLM_Model="pretrained-model-checkpoint"
   ```

2. **Run the Script**: Execute the script to train the model:
   
   ```bash
   python sentiment_analysis_with_lora.py
   ```

3. **Model Deployment**: After training, the model will be pushed to Hugging Face Hub with the name `samadpls/sentiment-analysis`.

## Results
The script will output sentiment predictions for a set of sample texts, showing the performance of both the untrained and trained models. Additionally, it calculates and reports the accuracy of the model on the validation dataset.

## License
This project is licensed under the MIT License.

---
