# AI-Driven Email Tone Modifier

## Project Overview
Our project aims to build an AI-powered assistant that rewrites emails in different tones (formal, casual, persuasive, apologetic, assertive) while preserving the original meaning. We are leveraging transformer-based models (T5, GPT-4) to enable automatic tone adaptation, enhancing communication effectiveness.

## 1. Data Collection & Preprocessing

🔹 Datasets Chosen:

- Enron Email Dataset (Business Emails - Formal, Casual, Persuasive)
- Hillary Clinton Email Dataset (Diplomatic & Formal Emails)
- Customer Service Email Dataset (Apologetic & Persuasive Emails)

🔹 Preprocessing Steps Completed:

- Removed email metadata (headers, timestamps, signatures)
- Normalized text (lowercasing, punctuation removal, contraction expansion)
- Tokenized the content for processing
- Merged all datasets into a unified preprocessed dataset

## 2. Model Development
🔹 Baseline Model:

As an initial step towards developing an effective tone rewriting system for transforming informal emails into their formal counterparts, a baseline model was implemented using a standard Sequence-to-Sequence (Seq2Seq) architecture based on Long Short-Term Memory (LSTM) networks.

The baseline model comprises an encoder-decoder framework, where both the encoder and decoder are constructed with LSTM layers and utilize randomly initialized word embeddings. The model was trained from scratch on a custom dataset consisting of parallel informal and formal email pairs. Key hyperparameters such as the number of LSTM layers and embedding dimensions were systematically varied to evaluate their impact on performance.

This baseline serves as a reference point for assessing the capabilities and limitations of traditional sequence models in the context of stylistic text rewriting. While such models are effective for capturing sequential dependencies, they lack the nuanced understanding and contextual awareness required for complex tasks like tone transformation. The results obtained using this model provide important insights into its performance bottlenecks and underscore the need for more advanced approaches incorporating attention mechanisms or pretrained language models.

🔹 Transformer-Based Models Selected:
- T5 (Text-to-Text Transfer Transformer) Fine-Tuning
- GPT-4 Fine-Tuning for Email Tone Adaptation

## 3. Model Training & Fine-Tuning
🔹 **Training Process of Baseline Model:**
**- Model Initialization**
The Seq2Seq model was constructed using an LSTM-based encoder and decoder. The encoder and decoder each received their own embedding layers initialized randomly. The encoder processed the input sequence to generate a context vector, which was then used by the decoder to generate the output sequence token by token. The input and output vocabulary sizes were determined from the training data.

**- Model Configuration**
The embedding dimension was set to 128, and the hidden state dimension of the LSTM layers was set to 256. Both the encoder and decoder used a single LSTM layer. The output of the decoder passed through a fully connected layer to project the hidden state to the vocabulary size of the target language.

**- Loss Function and Optimization**
The model was trained using the CrossEntropyLoss, with the ignore_index parameter set to the padding token index in order to exclude padding tokens from contributing to the loss. The Adam optimizer was used to update the model parameters.

**- Training Loop**
The model was trained for 100 epochs. During each epoch:

The model was set to training mode.

For each batch in the training dataset:

Input (src) and target (trg) sequences were moved to the appropriate device (CPU/GPU).

Gradients were reset using optimizer.zero_grad().

The model made predictions on the input sequence with the target sequence provided as input for teacher forcing.

The output sequence (excluding the first token, typically <sos>) was reshaped for compatibility with the loss function.

The target sequence was similarly reshaped.

The loss was computed and backpropagated.

The optimizer updated the model weights.

Batch loss was accumulated to compute epoch-level loss.

At the end of each epoch, the training loss was printed to monitor convergence.

**Next Step**
- Fine-tuning T5 for informal emaill conversion to formal emails.


## 4. Evaluation Metrics Defined
- BLEU Score - Measures similarity to human-written emails
- ROUGE Score - Evaluates text overlap with reference emails

**Baseline Model Evaluation**
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





