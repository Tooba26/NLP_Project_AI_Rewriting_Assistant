# AI-Driven Email Tone Modifier

## Project Overview
Our project aims to build an AI-powered assistant that rewrites emails in different tones (formal, casual, persuasive, apologetic, assertive) while preserving the original meaning. We are leveraging transformer-based models (T5, GPT-4) to enable automatic tone adaptation, enhancing communication effectiveness.

## 1. Data Collection & Preprocessing
- We intended to use the GYAFC dataset for this project and initiated the required approval process through Yahoo. After receiving the necessary authorization, we contacted the dataset authors via the email provided on their GitHub repository and forwarded the approval confirmation. However, we have not received any response from them to date.
- Then, We began by selecting two subsets from the Enron email dataset: one consisting of 1,000 general email sentences and another containing 5,000 shorter sentences, each ranging between 5 to 25 words in length.
- The email content was filtered to retain only the main message body, removing components such as headers, sender and receiver names, timestamps, and any other metadata not relevant to the message content.
- The remaining text underwent a cleaning process where unnecessary symbols were removed, all characters were converted to lowercase, and the content was standardized to ensure consistency and readability.
- In addition to the Enron data, a manually curated dataset of 1,000 sentence pairs was created. Each pair reflected a realistic transformation between an informal and a formal sentence, representative of tone differences commonly found in email communication.
- After cleaning, the three datasets were combined into a single dataset intended for model training.
- The initial combined dataset contained 6,893 rows and two columns, corresponding to informal and formal sentence pairs.
- A check for missing values confirmed there were none; however, duplicate entries were identified and removed, resulting in a final dataset size of 6,571 unique records.
- As a final preprocessing step, newline characters were eliminated, remaining special characters were removed, and the overall structure and format of the text were verified to ensure the dataset was clean and suitable for training.

## 2. Model Development
🔹 Baseline Model:

As an initial step towards developing an effective tone rewriting system for transforming informal emails into their formal counterparts, a baseline model was implemented using a standard Sequence-to-Sequence (Seq2Seq) architecture based on Long Short-Term Memory (LSTM) networks.

The baseline model comprises an encoder-decoder framework, where both the encoder and decoder are constructed with LSTM layers and utilize randomly initialized word embeddings. The model was trained from scratch on a custom dataset consisting of parallel informal and formal email pairs. Key hyperparameters such as the number of LSTM layers and embedding dimensions were systematically varied to evaluate their impact on performance.

This baseline serves as a reference point for assessing the capabilities and limitations of traditional sequence models in the context of stylistic text rewriting. While such models are effective for capturing sequential dependencies, they lack the nuanced understanding and contextual awareness required for complex tasks like tone transformation. The results obtained using this model provide important insights into its performance bottlenecks and underscore the need for more advanced approaches incorporating attention mechanisms or pretrained language models.

🔹 Transformer-Based Models Selected:
- T5 (Text-to-Text Transfer Transformer) Fine-Tuning
- Fine-tuning T5 for informal emaill conversion to formal emails.
Tone Transformations Handled:

Informal → Formal

Formal → Informal

Apologetic → Formal

Apologetic → Informal

Informal → Apologetic

Formal → Apologetic

## 3. Model Training & Fine-Tuning
🔹 **Training Process of Baseline Model:**

**- Model Initialization**
The Seq2Seq model was constructed using an LSTM-based encoder and decoder. The encoder and decoder each received their own embedding layers initialized randomly. The encoder processed the input sequence to generate a context vector, which was then used by the decoder to generate the output sequence token by token. The input and output vocabulary sizes were determined from the training data.

**- Model Configuration**
The embedding dimension was set to 128, and the hidden state dimension of the LSTM layers was set to 256. Both the encoder and decoder used a single LSTM layer. The output of the decoder passed through a fully connected layer to project the hidden state to the vocabulary size of the target language.

**- Loss Function and Optimization**
The model was trained using the CrossEntropyLoss, with the ignore_index parameter set to the padding token index in order to exclude padding tokens from contributing to the loss. The Adam optimizer was used to update the model parameters.

**- Training Loop**
The model was trained for 100 epochs. During each epoch the model was set to training mode.
For each batch in the training dataset, Input (src) and target (trg) sequences were moved to the appropriate device (CPU/GPU).
Gradients were reset using optimizer.zero_grad().
The model made predictions on the input sequence with the target sequence provided as input for teacher forcing.
The output sequence (excluding the first token, typically <sos>) was reshaped for compatibility with the loss function.
The target sequence was similarly reshaped. The loss was computed and backpropagated.
The optimizer updated the model weights.
Batch loss was accumulated to compute epoch-level loss.
At the end of each epoch, the training loss was printed to monitor convergence.

**Training of Fine-tuned T5 model**

Base Model: t5-base from HuggingFace.

Optimizer: AdamW

Learning Rate: 5e-5

Batch Size: 8

Training Setup:

Early stopping enabled based on validation loss.

Teacher forcing used during training.

Total epochs were 50 with early stopping based on tone transformation type.

The models were then pushed to HuggingFace models.

## 4. Evaluation Metrics Defined
- BLEU Score - Measures similarity to human-written emails
- ROUGE Score - Evaluates text overlap with reference emails

**Baseline Model Evaluation**

To evaluate the effect of different hyperparameter configurations on the performance of the baseline Seq2Seq model, several experiments were conducted by varying the number of LSTM layers, embedding dimension, and hidden size. The aim was to identify whether deeper models or larger embedding representations could improve the model's ability to convert informal emails into formal ones.

Despite the changes, the model's performance remained relatively low across all configurations, indicating the architectural limitations of LSTM-based Seq2Seq models for this task. The BLEU and ROUGE scores showed slight improvements with higher dimensions and 2–4 layers, but exact match accuracy remained 0.0 throughout, reflecting the model's inability to generate outputs that closely matched the reference formal sentences.
### 📈 Baseline Model

| Model              | BLEU Score | ROUGE-1 | ROUGE-2 | ROUGE-L | Exact Match |
|--------------------|------------|---------|---------|---------|--------------|
| Baseline Seq2Seq   | 0.0919     | 0.2378  | 0.1490  | 0.2243  | 0.0          |

### Experiments with different layers and embedding dimensions

| Embedding Dim | Hidden Dim | LSTM Layers | BLEU Score | ROUGE-1 | ROUGE-2 | ROUGE-L | Exact Match |
|---------------|------------|-------------|------------|---------|---------|---------|--------------|
| 128           | 256        | 1           | 0.0783     | 0.2141  | 0.1355  | 0.2047  | 0.0          |
| 128           | 256        | 2           | 0.0919     | 0.2378  | 0.1490  | 0.2243  | 0.0          |
| 128           | 256        | 4           | 0.0897     | 0.2293  | 0.1421  | 0.2179  | 0.0          |
| 128           | 256        | 8           | 0.0714     | 0.2015  | 0.1208  | 0.1926  | 0.0          |
| 256           | 512        | 2           | 0.1064     | 0.2512  | 0.1618  | 0.2399  | 0.0          |
| 256           | 512        | 4           | 0.0991     | 0.2467  | 0.1562  | 0.2345  | 0.0          |
| 256           | 512        | 8           | 0.0827     | 0.2195  | 0.1377  | 0.2102  | 0.0          |

### T5 model Performance
## 📊 Model Performance Summary

| Tone Transformation         | BLEU Score | ROUGE-1 | ROUGE-2 | ROUGE-L |
|------------------------------|------------|---------|---------|---------|
| Informal → Formal            | 0.8898     | 0.9482  | 0.9236  | 0.9482  |
| Apologetic → Informal        | 0.8874     | 0.9474  | 0.9256  | 0.9474  |
| Apologetic → Formal          | **0.9865** | **0.9992** | **0.9992** | **0.9992** |
| Informal → Apologetic        | 0.6437     | 0.7131  | 0.6517  | 0.7130  |
| Formal → Apologetic          | 0.6585     | 0.7213  | 0.6635  | 0.7211  |


## 5. Deployment
The models were loaded to HuggingFace to access for web app. The web app has been deployed on Streamlit. 
Can be accessed through following link:

https://email-tone-rewriter.streamlit.app

## 6. Key Observations
High Structural and Semantic Fidelity: The T5 model maintained the original intent while adjusting the tone.

Stronger Formalization: Formalization tasks (apologetic → formal) showed better results than emotional tone generation tasks.

Training Convergence: Loss curves showed consistent convergence without overfitting across all transformations.

## 7. Future Work
Explore T5-large or Flan-T5 models for improved performance.

Implement emotion conditioning to better handle nuanced emotional tones (e.g., apologetic, empathetic).

Extend tone options to assertive, persuasive, and empathetic styles.

Introduce affective computing modules for enhanced emotional tone detection and generation.







