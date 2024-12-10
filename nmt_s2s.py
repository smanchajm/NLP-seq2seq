from logging import Logger
import logging
import matplotlib.pyplot as plt
import sentencepiece  # type: ignore
import torch
import torch.nn as nn
from utils import preprocess, generate_predictions_file_for_submission
from trainutils import train
import torch.nn.functional as F


class Seq2SeqNMTwithAttention(nn.Module):
    def __init__(self):
        super().__init__()

        # BEGIN SOLUTION
        # Define model hyperparameters
        vocab_size = 8000
        embedding_dim = 256
        hidden_size = 256
        num_layers = 2
        dropout_rate = 0.5

        # Embedding layers initialized with random weights
        self.src_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.tgt_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Encoder: Bidirectional LSTM
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate,
            bidirectional=True,
            batch_first=False,
        )

        # Decoder: Unidirectional LSTM
        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=False,
        )

        # we initialize the hidden state of the decoder with a linear projection of the final hidden state of the encoder
        self.hidden_proj = nn.Linear(hidden_size * 2, hidden_size)

        # Project the decoder output to the vocabulary size
        self.output_proj = nn.Linear(hidden_size, vocab_size)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # END SOLUTION

    def encode(self, source):
        """Encode the source batch using a bidirectional LSTM encoder.

        Args:
          source: An integer tensor with shape (max_source_sequence_length,
            batch_size) containing subword indices for the source sentences.

        Returns:
          A tuple with three elements:
            encoder_output: The output of the bidirectional LSTM with shape
              (max_source_sequence_length, batch_size, 2 * hidden_size).
            encoder_mask: A boolean tensor with shape (max_source_sequence_length,
              batch_size) indicating which encoder outputs correspond to padding
              tokens. Its elements should be True at positions corresponding to
              padding tokens and False elsewhere.
            encoder_hidden: The final hidden states of the bidirectional LSTM (after
              a suitable projection) that will be used to initialize the decoder.
              This should be a pair of tensors (h_n, c_n), each with shape
              (num_layers, batch_size, hidden_size). Note that the hidden state
              returned by the LSTM cannot be used directly. Its initial dimension is
              twice the required size because it contains state from two directions.

        The first two return values are not required for the baseline model and will
        only be used later in the attention model. If desired, they can be replaced
        with None for the initial implementation.
        """

        # Implementation tip: consider using packed sequences to more easily work
        # with the variable-length sequences represented by the source tensor.
        # See https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.PackedSequence.

        # Implementation tip: there are many simple ways to combine the forward
        # and backward portions of the final hidden state, e.g. addition, averaging,
        # or a linear transformation of the appropriate size. Any of these
        # should let you reach the required performance.

        # length of each sequence in the batch (before padding)
        lengths = torch.sum(source != pad_id, axis=0)

        # YOUR CODE HERE
        # BEGIN SOLUTION

        # We project the source sequence to the embedding space
        embedded = self.src_embedding(source)

        # we pack the sequence to avoid unnecessary computation on padding tokens
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), enforce_sorted=False
        )

        # Run bidirectional encoder
        packed_output, (h_n, c_n) = self.encoder(packed)

        # We unpack the sequence output of the encoder
        encoder_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output)

        # Create encoder mask (True for padding)
        encoder_mask = source == 0

        # Process bidirectional hidden states
        h_n = torch.cat([h_n[0::2], h_n[1::2]], dim=-1)
        c_n = torch.cat([c_n[0::2], c_n[1::2]], dim=-1)

        # We project the final hidden states of the encoder to the decoder
        h_n = self.hidden_proj(h_n)
        c_n = self.hidden_proj(c_n)

        return encoder_output, encoder_mask, (h_n, c_n)
        # END SOLUTION

    def decode(self, decoder_input, initial_hidden, encoder_output, encoder_mask):
        """Run the decoder LSTM starting from an initial hidden state.

        The third and fourth arguments are not used in the baseline model, but are
        included for compatibility with the attention model in the next section.

        Args:
          decoder_input: An integer tensor with shape (max_decoder_sequence_length,
            batch_size) containing the subword indices for the decoder input. During
            evaluation, where decoding proceeds one step at a time, the initial
            dimension should be 1.
          initial_hidden: A pair of tensors (h_0, c_0) representing the initial
            state of the decoder, each with shape (num_layers, batch_size,
            hidden_size).
          encoder_output: The output of the encoder with shape
            (max_source_sequence_length, batch_size, 2 * hidden_size).
          encoder_mask: The output mask from the encoder with shape
            (max_source_sequence_length, batch_size). Encoder outputs at positions
            with a True value correspond to padding tokens and should be ignored.

        Returns:
          A tuple with three elements:
            logits: A tensor with shape (max_decoder_sequence_length, batch_size,
              vocab_size) containing unnormalized scores for the next-word
              predictions at each position.
            decoder_hidden: A pair of tensors (h_n, c_n) with the same shape as
              initial_hidden representing the updated decoder state after processing
              the decoder input.
            attention_weights: This will be implemented later in the attention
              model, but in order to maintain compatible type signatures, we also
              include it here. This can be None or any other placeholder value.
        """

        # We project the decoder input to the embedding space
        embedded = self.tgt_embedding(decoder_input)

        # Initialize lists to store logits and attention weights
        logits = []
        attention_weights_list = []

        # h and c are the hidden states of the decoder
        h, c = initial_hidden

        # Project encoder outputs to match decoder hidden size
        attention_proj = nn.Linear(2 * self.hidden_size, self.hidden_size).to(
            encoder_output.device
        )
        # Project context to match embedding size
        context_proj = nn.Linear(self.hidden_size, self.hidden_size).to(
            encoder_output.device
        )

        # We iterate over the sequence of the decoder input
        for t in range(embedded.size(0)):
            current_input = embedded[t].unsqueeze(0)

            # Project encoder outputs to match decoder hidden size
            projected_encoder_output = attention_proj(encoder_output)

            # We compute the alignment scores between the decoder hidden state and the encoder outputs
            alignment_scores = torch.bmm(
                projected_encoder_output.transpose(0, 1),
                h[-1].unsqueeze(2),
            ).squeeze(2)

            # We mask the alignment scores to avoid attending to padding tokens
            if encoder_mask is not None:
                alignment_scores = alignment_scores.masked_fill(
                    encoder_mask.transpose(0, 1), float("-inf")
                )

            # Wz compute attention weights using softmax
            attention_weights = F.softmax(alignment_scores, dim=1)
            attention_weights_list.append(attention_weights)

            # we compute context as the weighted sum of the encoder outputs
            context = torch.bmm(
                attention_weights.unsqueeze(1),
                projected_encoder_output.transpose(0, 1),
            ).transpose(0, 1)

            # Project context to match embedding size
            context = context_proj(context)

            decoder_input_with_context = torch.cat([current_input, context], dim=-1)

            decoder_input_processed = nn.Linear(
                decoder_input_with_context.size(-1), self.hidden_size
            ).to(decoder_input_with_context.device)(decoder_input_with_context)

            # Run decoder step
            decoder_output, (h, c) = self.decoder(decoder_input_processed, (h, c))

            current_logits = self.output_proj(decoder_output)
            logits.append(current_logits)

        # We concatenate the logits and attention weights along the sequence dimension
        logits = torch.cat(logits, dim=0)
        attention_weights = torch.stack(attention_weights_list, dim=0)

        return logits, (h, c), attention_weights

    def compute_loss(self, source, target):
        """Run the model on the source and compute the loss on the target.

        Args:
          source: An integer tensor with shape (max_source_sequence_length,
            batch_size) containing subword indices for the source sentences.
          target: An integer tensor with shape (max_target_sequence_length,
            batch_size) containing subword indices for the target sentences.

        Returns:
          A scalar float tensor representing cross-entropy loss on the current batch.
        """

        # Implementation tip: don't feed the target tensor directly to the decoder.
        # To see why, note that for a target sequence like <s> A B C </s>, you would
        # want to run the decoder on the prefix <s> A B C and have it predict the
        # suffix A B C </s>.

        # YOUR CODE HERE
        ...

        # BEGIN SOLUTION
        # We encode the source sequence to get the encoder output, mask, and hidden states
        encoder_output, encoder_mask, encoder_hidden = self.encode(source)

        decoder_input = target[:-1]
        decoder_target = target[1:]

        # We decode the target sequence to get the logits
        logits, _, _ = self.decode(
            decoder_input, encoder_hidden, encoder_output, encoder_mask
        )

        # Flatten logits and target
        logits = logits.view(-1, logits.size(-1))
        decoder_target = decoder_target.view(-1)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, decoder_target, ignore_index=0)

        return loss

        # END SOLUTION


if __name__ == "__main__":
    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configurer le logger
    logging.basicConfig(
        level=logging.DEBUG,  # Niveau minimum des messages à afficher
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Format du message
        datefmt="%Y-%m-%d %H:%M:%S",  # Format de la date
    )

    # Créer un logger
    logger = logging.getLogger("Main")

    logger.info(f"Début du programme {device}")

    # Define the paths to your local data files
    train_src_path = "multi30k/train.en"
    train_tgt_path = "multi30k/train.de"
    validation_src_path = "multi30k/val.en"
    validation_tgt_path = "multi30k/val.de"
    test_src_path = "multi30k/test.en"
    test_tgt_path = "multi30k/test.de"

    # Read the data from the files
    with open(train_src_path, "r", encoding="utf-8") as f_src, open(
        train_tgt_path, "r", encoding="utf-8"
    ) as f_tgt:
        src_sentences = f_src.readlines()
        tgt_sentences = f_tgt.readlines()

    with open(validation_src_path, "r", encoding="utf-8") as f_src, open(
        validation_tgt_path, "r", encoding="utf-8"
    ) as f_tgt:
        val_src_sentences = f_src.readlines()
        val_tgt_sentences = f_tgt.readlines()

    with open(test_src_path, "r", encoding="utf-8") as f_src, open(
        test_tgt_path, "r", encoding="utf-8"
    ) as f_tgt:
        test_src_sentences = f_src.readlines()
        test_tgt_sentences = f_tgt.readlines()

    # Preprocess the data and create aligned sentence pairs
    training_data = []
    for src_line, tgt_line in zip(src_sentences, tgt_sentences):
        source = preprocess(src_line)
        target = preprocess(tgt_line)
        training_data.append((source, target))

    validation_data = []
    for src_line, tgt_line in zip(val_src_sentences, val_tgt_sentences):
        source = preprocess(src_line)
        target = preprocess(tgt_line)
        validation_data.append((source, target))

    test_data = []
    for src_line, tgt_line in zip(test_src_sentences, test_tgt_sentences):
        source = preprocess(src_line)
        target = preprocess(tgt_line)
        test_data.append((source, target))

    # Prepare for vocabulary
    vocab = sentencepiece.SentencePieceProcessor()
    vocab.Load("multi30k.model")
    print("Vocabulary size:", vocab.GetPieceSize())
    pad_id = vocab.PieceToId("<pad>")
    bos_id = vocab.PieceToId("<s>")
    eos_id = vocab.PieceToId("</s>")

    # You are welcome to adjust these parameters based on your model implementation.
    num_epochs = 10
    batch_size = 64

    # Create the model
    model = Seq2SeqNMTwithAttention().to(device)
    train(model, num_epochs, batch_size, "nmt_model.pt", training_data, validation_data)

    # Charger les poids sauvegardés
    # model.load_state_dict(torch.load("nmt_model.pt", weights_only=True))
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Generate test set predictions
    student_id = "49005153"
    generate_predictions_file_for_submission(
        f"results/{student_id}_predictions.json", model, test_data, 0.0
    )

    # YOUR CODE HERE
    def visualize_attention(model, source_sentence, target_sentence, vocab, save_path):
        """Visualize the attention weights of the model for a source-target pair.

         we can understand the plot as follows:
        - The x-axis represents the source sentence.
        - The y-axis represents the target sentence.
        - The plot shows how much attention the model pays to each source word when predicting each target word.

        Args:
          model: A sequence-to-sequence model with attention.
          source_sentence: A string containing the source sentence.
          target_sentence: A string containing the target sentence.
          vocab: A SentencePiece processor.
          save_path: A string path to save the plot.
        """
        source_tensor = (
            torch.tensor([vocab.EncodeAsIds(source_sentence)], dtype=torch.long)
            .to(device)
            .T
        )
        target_tensor = (
            torch.tensor([vocab.EncodeAsIds(target_sentence)], dtype=torch.long)
            .to(device)
            .T
        )

        # Compute attention weights by running the model
        encoder_output, encoder_mask, encoder_hidden = model.encode(source_tensor)
        decoder_input = target_tensor[:-1]
        _, _, attention_weights = model.decode(
            decoder_input, encoder_hidden, encoder_output, encoder_mask
        )

        attention_weights = attention_weights.squeeze(1).cpu().detach().numpy()

        # Plot the attention weights as a heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(attention_weights, cmap="viridis")
        plt.xticks(
            range(len(source_sentence.split())), source_sentence.split(), rotation=90
        )
        plt.yticks(range(len(target_sentence.split())), target_sentence.split())
        plt.colorbar(label="Attention Weight")
        plt.title("Attention Map")
        plt.savefig(save_path)
        plt.show()
        print("Source sentence:", source_sentence)

    # Visualize attention for two sentence pairs from validation data
    # You can adjust index of validation_data to visualize different sentence pairs
    source_sentence1, target_sentence1 = validation_data[0]
    source_sentence2, target_sentence2 = validation_data[1]

    visualize_attention(
        model,
        " ".join(source_sentence1),
        " ".join(target_sentence1),
        vocab,
        f"results/49005153_attention_1.png",
    )
    visualize_attention(
        model,
        " ".join(source_sentence2),
        " ".join(target_sentence2),
        vocab,
        f"results/49005153_attention_2.png",
    )
    # END SOLUTION
