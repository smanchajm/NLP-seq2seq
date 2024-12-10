import string
import logging
import torch
import sacrebleu  # type: ignore
import json
import sentencepiece  # type: ignore
import traceback
import torch.nn.functional as F


#######################
### For IBM Model 1 ###
#######################


def preprocess(sentence):
    sentence = sentence.translate(
        str.maketrans("", "", string.punctuation)
    )  # strip punctuation
    return sentence.strip().lower().split()


# def visualize_alignment(alignment, source_sentence, target_sentence):
#     # Create an alignment matrix
#     alignment_matrix = [[" " for _ in target_sentence] for _ in source_sentence]
#     for i, j in alignment:
#         alignment_matrix[i][j] = "✔"

#     # Print the matrix with source and target words
#     print("Alignment Matrix:")
#     print("      " + "  ".join(f"{t:>5}" for t in target_sentence))
#     for i, row in enumerate(alignment_matrix):
#         print(f"{source_sentence[i]:>5} " + "  ".join(f"{cell:>5}" for cell in row))


def visualize_alignment(alignment, source_sentence, target_sentence):
    """
    Modified version of the visualize_alignment function to display the alignment matrix with aligned source and target words.

    Args:
        alignment (list of tuple): The alignment between source and target words.
        source_sentence (list of str): The source sentence.
        target_sentence (list of str): The target sentence.
    """

    # we compute the maximum length to have a fixed width
    max_source_word_length = max(len(word) for word in source_sentence)
    max_target_word_length = max(len(word) for word in target_sentence)
    cell_width = max(
        max_source_word_length, max_target_word_length, 5
    )  # minimum width of 5

    alignment_matrix = [[" " for _ in target_sentence] for _ in source_sentence]
    for i, j in alignment:
        alignment_matrix[i][j] = "✔"

    # Print the matrix with source and target words
    print("Alignment Matrix:")
    print(
        " " * (cell_width + 1)
        + "  ".join(f"{t:>{cell_width}}" for t in target_sentence)
    )
    for i, row in enumerate(alignment_matrix):
        print(
            f"{source_sentence[i]:>{cell_width}} "
            + "  ".join(f"{cell:>{cell_width}}" for cell in row)
        )


#######################
### For Seq2Seq NMT ###
#######################


# vocaburary and essential definitions
# you can make this part as comment for SMT implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = sentencepiece.SentencePieceProcessor()
vocab.Load("multi30k.model")
pad_id = vocab.PieceToId("<pad>")
bos_id = vocab.PieceToId("<s>")
eos_id = vocab.PieceToId("</s>")


def predict_greedy(model, sentences, max_length=100):
    # TODO: implement greedy search
    pass


def predict_beam(model, sentences, beam_width=5, max_length=100):
    """
    implement beam search to generate translations from a Seq2Seq model.

    Args:
        model (Seq2SeqNMTwithAttention): The Seq2Seq model with attention.
        sentences (list of list of str): The source sentences (lists of words as text).
        beam_width (int): The width of the beam for the search.
        max_length (int): The maximum length of the generated sequences.

    Returns:
        list of list of str: The translated sentences in text.
    """

    model.eval()
    results = []

    with torch.no_grad():
        sentence_id = 0

        # We iterate over all the sentences of the dataset
        for sentence in sentences:
            sentence_id += 1

            # Xe convert the sentence to a tensor
            src_tensor = torch.tensor(
                [vocab.PieceToId(word) for word in sentence],
                dtype=torch.long,
                device=device,
            ).unsqueeze(1)

            # We encode the source sentence to get the encoder hidden states
            encoder_output, encoder_mask, encoder_hidden = model.encode(src_tensor)

            # We initialize the beam search with the bos token <s>
            beams = [(torch.tensor([bos_id]).to(device), 0.0, encoder_hidden)]
            completed_sequences = []

            # We iterate over the maximum length of the sequence
            for _ in range(max_length):
                new_beams = []

                # We iterate over the beams (k beams)
                for seq, score, hidden in beams:

                    # if the sequence is completed, we add it to the completed sequences
                    if seq[-1] == eos_id:
                        completed_sequences.append((seq, score))
                        continue

                    # We decode (generate the next token) using the last token of the sequence
                    logits, hidden, _ = model.decode(
                        seq[-1].unsqueeze(0).unsqueeze(1),
                        hidden,
                        encoder_output,
                        encoder_mask,
                    )
                    # We squeeze the logits to remove the batch and sequence dimensions
                    logits = logits.squeeze(0).squeeze(0)
                    topk_probs, topk_indices = torch.topk(
                        F.log_softmax(logits, dim=-1), beam_width
                    )

                    # We iterate over the top k probabilities
                    for prob, idx in zip(topk_probs, topk_indices):
                        new_seq = torch.cat([seq, idx.unsqueeze(0)])
                        new_score = score + prob.item()
                        new_beams.append((new_seq, new_score, hidden))

                # We sort the new beams by score and keep the top k beams
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

                # We check if we have enough completed sequences
                if len(completed_sequences) >= beam_width:
                    break

            # We get the best sequence
            if len(completed_sequences) > 0:
                best_seq, _ = max(completed_sequences, key=lambda x: x[1])
            else:
                best_seq, _, _ = beams[0]

            # We decode the best sequence to words and add it to the results
            decoded_sentence = [
                vocab.IdToPiece(idx.item())
                for idx in best_seq
                if idx.item() not in {bos_id, eos_id, pad_id}
            ]
            results.append(decoded_sentence)

    return results


def evaluate(model, dataset, batch_size=64, method="beam"):
    """
    evaluate is a function that takes a model, a dataset and a method to generate predictions
    and returns the BLEU score for the dataset

    Args:
        model (Seq2SeqNMTwithAttention): The Seq2Seq model with attention.
        dataset (list of tuple): The dataset to generate predictions for.
        batch_size (int): The batch size to use for prediction.
        method (str): The decoding method to use. Either "greedy" or "beam".

    Returns:
        float: The BLEU score for the dataset.

    """

    assert method in {"greedy", "beam"}
    source_sentences = [example[0] for example in dataset]
    target_sentences = [" ".join(example[1]) for example in dataset]
    model.eval()
    predictions = []
    with torch.no_grad():
        for start_index in range(0, len(source_sentences), batch_size):
            if method == "beam":
                prediction_batch = predict_beam(
                    model, source_sentences[start_index : start_index + batch_size]
                )
            else:
                prediction_batch = predict_beam(
                    model, source_sentences[start_index : start_index + batch_size]
                )
                prediction_batch = [candidates[0] for candidates in prediction_batch]
            predictions.extend(prediction_batch)
    return sacrebleu.corpus_bleu(predictions, [target_sentences]).score


def get_raw_predictions(model, dataset, method="beam", batch_size=64):
    """
    get raw prediction is a function that takes a model, a dataset and a method to generate predictions
    and returns the predictions in a list

    Args:
        model (Seq2SeqNMTwithAttention): The Seq2Seq model with attention.
        dataset (list of tuple): The dataset to generate predictions for.
        method (str): The decoding method to use. Either "greedy" or "beam".
        batch_size (int): The batch size to use for prediction.

    Returns:
        list of str: The raw predictions for the dataset.

    """
    assert method in {"greedy", "beam"}
    source_sentences = [example[0] for example in dataset]
    model.eval()
    predictions = []
    with torch.no_grad():
        for start_index in range(0, len(source_sentences), batch_size):
            try:
                if method == "greedy":
                    print(
                        f"Calling predict_greedy for batch {start_index} to {start_index + batch_size}"
                    )
                    prediction_batch = predict_beam(
                        model, source_sentences[start_index : start_index + batch_size]
                    )
                else:
                    print(
                        f"Calling predict_beam for batch {start_index} to {start_index + batch_size}"
                    )
                    prediction_batch = predict_beam(
                        model, source_sentences[start_index : start_index + batch_size]
                    )
                predictions.extend(prediction_batch)
            except Exception as e:
                print(f"Error during batch {start_index}: {e}")
                import traceback

                traceback.print_exc()
                predictions.extend(
                    [None] * (start_index + batch_size - len(source_sentences))
                )
    return predictions


def generate_predictions_file_for_submission(filepath, model, dataset, bleu_score):

    # Configurer le logger
    logging.basicConfig(
        level=logging.DEBUG,  # Niveau minimum des messages à afficher
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Format du message
        datefmt="%Y-%m-%d %H:%M:%S",  # Format de la date
    )

    # Créer un logger
    logger2 = logging.getLogger("predictor")

    models = {"attention": model}
    datasets = {"test": dataset}
    methods = ["beam"]  # you can adjust according to your implementation
    predictions = {}

    for model_name, model in models.items():
        line_number: int = 0
        total_lines: int = len(dataset)
        for dataset_name, dataset in datasets.items():
            logger2.info(f"Line number: {line_number}, Total lines: {total_lines}")
            for method in methods:
                print(
                    "Getting predictions for {} model on {} set using {} "
                    "search...".format(model_name, dataset_name, method)
                )
                if model_name not in predictions:
                    predictions[model_name] = {}
                if dataset_name not in predictions[model_name]:
                    predictions[model_name][dataset_name] = {}
                try:
                    predictions[model_name][dataset_name][method] = get_raw_predictions(
                        model, dataset, method
                    )
                except Exception as e:
                    print(f"!!! WARNING: An exception was raised: {e} !!!")
                    print(f"Traceback details:")

                    traceback.print_exc()
                    predictions[model_name][dataset_name][method] = None
    print("Writing predictions to {}...".format(filepath))
    with open(filepath, "w", encoding="utf-8") as outfile:
        json.dump(predictions, outfile, indent=2, ensure_ascii=False)
        # Write BLEU score to the end of the file
        outfile.write("\n")
        outfile.write("BLEU: {:.2f}".format(bleu_score))

    print("Finished writing predictions to {}.".format(filepath))
