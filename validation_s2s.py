import torch
from nmt_s2s import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Prepare for vocabulary
    vocab = sentencepiece.SentencePieceProcessor()
    vocab.Load("multi30k.model")
    print("Vocabulary size:", vocab.GetPieceSize())
    pad_id = vocab.PieceToId("<pad>")
    bos_id = vocab.PieceToId("<s>")
    eos_id = vocab.PieceToId("</s>")

    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the paths to your local data files
    train_src_path = "multi30k/train.en"
    train_tgt_path = "multi30k/train.de"
    validation_src_path = "multi30k/val.en"
    validation_tgt_path = "multi30k/val.de"
    test_src_path = "multi30k/test.en"
    test_tgt_path = "multi30k/test.de"

    with open(validation_src_path, "r", encoding="utf-8") as f_src, open(
        validation_tgt_path, "r", encoding="utf-8"
    ) as f_tgt:
        val_src_sentences = f_src.readlines()
        val_tgt_sentences = f_tgt.readlines()

    validation_data = []
    for src_line, tgt_line in zip(val_src_sentences, val_tgt_sentences):
        source = preprocess(src_line)
        target = preprocess(tgt_line)
        validation_data.append((source, target))

    # Instancier le modèle (structure identique à celle utilisée lors de la sauvegarde)
    model = Seq2SeqNMTwithAttention().to(device)

    # Charger les poids sauvegardés
    model.load_state_dict(torch.load("nmt_model.pt"))

    def visualize_attention(model, source_sentence, target_sentence, vocab, save_path):
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

        encoder_output, encoder_mask, encoder_hidden = model.encode(source_tensor)
        decoder_input = target_tensor[:-1]
        logits, _, attention_weights = model.decode(
            decoder_input, encoder_hidden, encoder_output, encoder_mask
        )

        attention_weights = attention_weights.squeeze(1).cpu().detach().numpy()
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

    # Visualize attention for two sentence pairs from validation data
    # You can adjust index of validation_data to visualize different sentence pairs
    source_sentence1, target_sentence1 = validation_data[0]
    source_sentence2, target_sentence2 = validation_data[1]

    visualize_attention(
        model,
        " ".join(source_sentence1),
        " ".join(target_sentence1),
        vocab,
        f"results/0000_attention_1.png",
    )
    visualize_attention(
        model,
        " ".join(source_sentence2),
        " ".join(target_sentence2),
        vocab,
        f"results/{student_id}_attention_2.png",
    )
