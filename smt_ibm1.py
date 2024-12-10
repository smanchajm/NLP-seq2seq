import math
import numpy as np
import pickle
import tqdm
from itertools import product
from utils import preprocess
from utils import visualize_alignment


class IBMModel1:
    def __init__(self, data, num_iterations=10, epsilon=1.0, compute_perplexity=True):
        self.data = data  # aligned corpus as shown above
        self.num_iterations = num_iterations  # iterations of expectation-maximization
        self.epsilon = epsilon
        self.compute_perplexity = compute_perplexity

        # Preprocess bitext data:
        self.source_words, self.target_words = set(), set()
        for source, target in self.data:
            self.source_words.update(source)
            self.target_words.update(target)

        # Initialize uniform probabilities:
        self.translation_probs = {
            (s, t): 1.0 / len(self.target_words)
            for s, t in product(self.source_words, self.target_words)
        }

    def e_step(self):
        """
        Compute the expected counts of word translations and the total counts of source words.

        Returns:
            counts: A dictionary with keys (source, target) and values the expected count of the translation.
            total_s: A dictionary with keys source and values the total count of the source word.
        """

        # We will store the expected counts of word translations and the total counts of source words
        counts = {(s, t): 0 for s, t in self.translation_probs}
        total_s = {s: 0 for s in self.source_words}

        for source, target in self.data:
            # Compute normalization factor for each target word
            for t in target:
                total_prob = sum(self.translation_probs[(s, t)] for s in source)
                for s in source:
                    contribution = self.translation_probs[(s, t)] / total_prob
                    counts[(s, t)] += contribution
                    total_s[s] += contribution

        return counts, total_s

    def m_step(self, counts, total_s):
        """
        Update the translation probabilities using the expected counts.

        Args:
            counts: A dictionary with keys (source, target) and values the expected count of the translation.
            total_s: A dictionary with keys source and values the total count of the source word.
        """

        for (s, t), count in counts.items():
            self.translation_probs[(s, t)] = count / total_s[s]

    def train(self):
        """
        Train the IBM Model 1 using the EM algorithm.
        """

        # Run the EM algorithm for num_iterations
        for idx in tqdm.tqdm(range(self.num_iterations)):
            if self.compute_perplexity:
                print("Iteration: {} | Perplexity: {}".format(idx, self.perplexity()))
            counts, total = self.e_step()
            self.m_step(counts, total)
        if self.compute_perplexity:
            print(
                "Iteration: {} | Perplexity: {}".format(
                    self.num_iterations, self.perplexity()
                )
            )

    def probability(self, source, target):
        """
        Compute the probability of target given source using IBM Model 1 formula, including epsilon.

        Args:
            source: list of source words.
            target: list of target words.

        Returns:
            prob: The probability of target given source.
        """

        target_length = len(target)  # Length of the target sentence
        source_length = len(source)

        # Add a NULL token for alignment
        source = ["NULL"] + source

        # Use epsilon normalization
        prob = self.epsilon / ((source_length + 1) ** target_length)

        # We compute the probability of the target sentence given the source sentence
        for word_t in target:
            align_prob_sum = 0
            for word_s in source:
                align_prob_sum += self.translation_probs.get((word_s, word_t), 0)
            prob *= align_prob_sum

        return prob

    def perplexity(self):
        log_sum = 0
        total_words = 0

        for source, target in self.data:
            prob = self.probability(source, target)
            log_sum += math.log(prob)
            total_words += len(target)

        return math.exp(-log_sum / total_words)

    def get_alignment(self, source, target):
        """
        Compute the alignment between source and target sentences using IBM Model

        Args:
            source: list of source words.
            target: list of target words.

        Returns:
            alignment: A list of tuples with the alignment pairs.
        """

        alignment = []
        for t_idx, t in enumerate(target):
            best_prob = 0
            best_s_idx = None
            for s_idx, s in enumerate(source):
                prob = self.translation_probs[(s, t)]
                if prob > best_prob:
                    best_prob = prob
                    best_s_idx = s_idx
            if best_s_idx is not None:
                alignment.append((best_s_idx, t_idx))
        return alignment


if __name__ == "__main__":
    # Define the paths to your local data files
    train_src_path = "multi30k/train.en"
    train_tgt_path = "multi30k/train.de"

    # Read the data from the files
    with open(train_src_path, "r", encoding="utf-8") as f_src, open(
        train_tgt_path, "r", encoding="utf-8"
    ) as f_tgt:
        src_sentences = f_src.readlines()
        tgt_sentences = f_tgt.readlines()

    # Preprocess the data and create aligned sentence pairs
    aligned_data = []
    for src_line, tgt_line in zip(src_sentences[:1000], tgt_sentences[:1000]):
        source = preprocess(src_line)
        target = preprocess(tgt_line)
        aligned_data.append((source, target))

    # Train the IBM Model 1
    ibm = IBMModel1(aligned_data, compute_perplexity=False)
    ibm.train()

    # # Visualize the alignment for a sample sentence pair
    # sample_source = aligned_data[0][0]  # Première phrase source
    # sample_target = aligned_data[0][1]  # Première phrase cible

    # # Obtenir les alignements avec la méthode get_alignment
    # sample_alignment = ibm.get_alignment(sample_source, sample_target)

    # Appeler la fonction visualize_alignment pour 5 alignements aléatoires
    for i in range(15):
        sample_idx = np.random.randint(len(aligned_data))
        sample_source, sample_target = aligned_data[sample_idx]
        sample_alignment = ibm.get_alignment(sample_source, sample_target)
        visualize_alignment(sample_alignment, sample_source, sample_target)

    # visualize_alignment(sample_alignment, sample_source, sample_target)

    pass
    ## End of implementation

    # Save the translation probabilities
    student_id = "49005153"
    with open(f"results/smt_{student_id}.pkl", "wb") as outfile:
        pickle.dump(ibm.translation_probs, outfile, protocol=pickle.HIGHEST_PROTOCOL)
