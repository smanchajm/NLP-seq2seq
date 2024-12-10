#######################
### For Seq2Seq NMT ###
#######################
import torch
import tqdm
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece  # type: ignore

# vocaburary and essential definitions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = sentencepiece.SentencePieceProcessor()
vocab.Load("multi30k.model")
pad_id = vocab.PieceToId("<pad>")
bos_id = vocab.PieceToId("<s>")
eos_id = vocab.PieceToId("</s>")


def train(model, num_epochs, batch_size, model_file, training_data, validation_data):
    """Train the model and save its best checkpoint.

    Model performance across epochs is evaluated using token-level accuracy on the
    validation set. The best checkpoint obtained during training will be stored on
    disk and loaded back into the model at the end of training.
    """
    optimizer = torch.optim.Adam(model.parameters())
    best_accuracy = 0.0
    for epoch in tqdm.trange(num_epochs, desc="training", unit="epoch"):
        with tqdm.tqdm(
            make_batch_iterator(training_data, batch_size, shuffle=True),
            desc="epoch {}".format(epoch + 1),
            unit="batch",
            total=math.ceil(len(training_data) / batch_size),
        ) as batch_iterator:
            model.train()
            total_loss = 0.0
            for i, (source, target) in enumerate(batch_iterator, start=1):
                optimizer.zero_grad()
                loss = model.compute_loss(source, target)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                batch_iterator.set_postfix(mean_loss=total_loss / i)
            validation_perplexity, validation_accuracy = evaluate_next_token(
                model, validation_data
            )
            batch_iterator.set_postfix(
                mean_loss=total_loss / i,
                validation_perplexity=validation_perplexity,
                validation_token_accuracy=validation_accuracy,
            )
            if validation_accuracy > best_accuracy:
                print(
                    "Obtained a new best validation accuracy of {:.2f}, saving model "
                    "checkpoint to {}...".format(validation_accuracy, model_file)
                )
                torch.save(model.state_dict(), model_file)
                best_accuracy = validation_accuracy
    print("Reloading best model checkpoint from {}...".format(model_file))
    model.load_state_dict(
        torch.load(
            model_file,
        )
    )


def evaluate_next_token(model, dataset, batch_size=64):
    """Compute token-level perplexity and accuracy metrics.

    Note that the perplexity here is over subwords, not words.

    This function is used for validation set evaluation at the end of each epoch
    and should not be modified.
    """
    model.eval()
    total_cross_entropy = 0.0
    total_predictions = 0
    correct_predictions = 0
    with torch.no_grad():
        for source, target in make_batch_iterator(dataset, batch_size, shuffle=False):
            encoder_output, encoder_mask, encoder_hidden = model.encode(source)
            decoder_input, decoder_target = target[:-1], target[1:]
            logits, decoder_hidden, attention_weights = model.decode(
                decoder_input, encoder_hidden, encoder_output, encoder_mask
            )
            total_cross_entropy += F.cross_entropy(
                logits.permute(1, 2, 0),
                decoder_target.permute(1, 0),
                ignore_index=0,
                reduction="sum",
            ).item()  # ignore padding index
            total_predictions += (decoder_target != 0).sum().item()
            correct_predictions += (
                ((decoder_target != 0) & (decoder_target == logits.argmax(2)))
                .sum()
                .item()
            )
    perplexity = math.exp(total_cross_entropy / total_predictions)
    accuracy = 100 * correct_predictions / total_predictions
    return perplexity, accuracy


def make_batch(sentences):
    """
    To do: Implement the make_batch function.
    Args:
      sentences: A list of strings, where each string represents a sentence.

    Yield:
      A torch tensor of shape (max_sequence_length, batch_size) containing the
      subword indices for the input sentences. The max_sequence_length should be
      the length of the longest sentence in the batch, and shorter sentences
      should be padded with the <pad> token. The batch_size should be the number
      of sentences in the batch. The tensor should be located on the device
      defined at the beginning.

    Hint:
      vocab.EncodeAsIds(sentence) returns a list of subword indices for a
      sentence, and nn.utils.rnn.pad_sequence can be used to pad the sequences.
    """
    # Encode sentences to lists of subword indices
    encoded_sentences = [vocab.EncodeAsIds(sentence) for sentence in sentences]

    # Convert lists to tensors
    tensor_sentences = [
        torch.tensor(sentence, dtype=torch.long) for sentence in encoded_sentences
    ]

    # Pad the sequences to the same length
    padded_tensor = nn.utils.rnn.pad_sequence(tensor_sentences, padding_value=pad_id)

    # Move tensor to the specified device
    padded_tensor = padded_tensor.to(device)

    return padded_tensor


def make_batch_iterator(dataset, batch_size, shuffle=False):
    """Make a batch iterator that yields source-target pairs.

    Args:
      dataset: A torchtext dataset object.
      batch_size: An integer batch size.
      shuffle: A boolean indicating whether to shuffle the examples.

    Yields:
      Pairs of tensors constructed by calling the make_batch function on the
      source and target sentences in the current group of examples. The max
      sequence length can differ between the source and target tensor, but the
      batch size will be the same. The final batch may be smaller than the given
      batch size.
    """

    examples = list(dataset)
    if shuffle:
        random.shuffle(examples)

    for i in range(0, len(examples), batch_size):
        batch_examples = examples[i : i + batch_size]
        source_sentences = [" ".join(ex[0]) for ex in batch_examples]
        target_sentences = [" ".join(ex[1]) for ex in batch_examples]

        source_tensor = make_batch(source_sentences)
        target_tensor = make_batch(target_sentences)

        yield (source_tensor, target_tensor)
