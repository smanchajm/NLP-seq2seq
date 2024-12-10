import sentencepiece
from utils import preprocess

if __name__ == "__main__":
    # Define the paths to your local data files
    train_src_path = 'multi30k/train.de'
    train_tgt_path = 'multi30k/train.en'

    # Read the data from the files
    with open(train_src_path, 'r', encoding='utf-8') as f_src, open(train_tgt_path, 'r', encoding='utf-8') as f_tgt:
        src_sentences = f_src.readlines()
        tgt_sentences = f_tgt.readlines()

    # Preprocess the data and create aligned sentence pairs
    training_data = []
    for src_line, tgt_line in zip(src_sentences, tgt_sentences):
        source = preprocess(src_line)
        target = preprocess(tgt_line)
        training_data.append((source, target))
    
    print("Number of training examples:", len(training_data))

    args = {
        "pad_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "unk_id": 3,
        "input": "multi30k/train.de,multi30k/train.en",
        "vocab_size": 8000,
        "model_prefix": "multi30k",
    }
    combined_args = " ".join(
        "--{}={}".format(key, value) for key, value in args.items())
    sentencepiece.SentencePieceTrainer.Train(combined_args)

    # This creates two files: multi30k.model and multi30k.vocab. 
    # The first is a binary file containing the relevant data for the vocabulary. 
    # The second is a human-readable listing of each subword and its associated score.
    # Please do not modify the arguments above.