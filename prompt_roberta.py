import os  # noqa
import random  # noqa

import torch  # noqa
from torch import Tensor  # noqa
from transformers import (
    RobertaForSequenceClassification,  # noqa
    RobertaTokenizer,
)

# Define emotion labels
EMOTION_LABELS = [
    "anger",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "disgust",
]

NUM_ANSWERS = 5
PROMPT_EXAMPLES = {
    "My pride hurt worse than my leg did.": ["anger", "sadness"],
    "Now my parents live in the foothills, and the college is in a large valley.": [
        "none"
    ],
    "I still cannot explain this.": ["fear", "surprise"],
    "Then I decided to try and get up to go to the restroom, but I couldn't move!": [
        "fear",
        "surprise",
    ],
}

# Define threshold for binary classification
THRESHOLD = 0.5


DEBUG_PRINT_ALL_PROBABILITIES = False

random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# MODEL_PATH = "models/roberta-base/"
MODEL_PATH = "output/roberta-semeval/"
TOKENIZER_PATH = "models/roberta-base/"


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model {MODEL_PATH} not found.")


def prompt():
    models = []

    for i in range(NUM_ANSWERS):
        # Load model and tokenizer
        tokenizer = RobertaTokenizer.from_pretrained(
            TOKENIZER_PATH, cache_dir="cache-dir/"
        )
        model = RobertaForSequenceClassification.from_pretrained(
            MODEL_PATH,
            num_labels=len(EMOTION_LABELS),
            cache_dir="cache-dir/",
            ignore_mismatched_sizes=True,
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        # TODO: instead of always loading the same model, load different models each iteration

        # Set model to evaluation mode to disable dropout
        model.eval()

        models.append((model, tokenizer))

    for prompt, answer in PROMPT_EXAMPLES.items():
        print(" ")
        print("-" * 50)
        print(f"Prompt: {prompt}")
        print(" ")

        voting_table = {}
        for emotion in EMOTION_LABELS:
            voting_table[emotion] = 0

        # FIXME: for now the model just gives deterministically the same output each run, except when i load the model each iteration

        for i in range(NUM_ANSWERS):
            print("Run:", i + 1)

            model, tokenizer = models[i]

            # Tokenize the input
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, padding=True
            ).to(model.device)

            # Perform inference
            with torch.no_grad():
                outputs: Tensor = model(**inputs)

            # Get predicted probabilities
            probabilities = torch.sigmoid(outputs.logits)

            # Apply threshold
            predicted_labels = (probabilities > THRESHOLD).int().squeeze().tolist()

            # Get predicted emotions
            predicted_emotions = [
                EMOTION_LABELS[j] for j, val in enumerate(predicted_labels) if val == 1
            ]

            for emotion in predicted_emotions:
                voting_table[emotion] += 1

            if DEBUG_PRINT_ALL_PROBABILITIES:
                print("Probabilities:")
                for label, prob in zip(
                    EMOTION_LABELS, probabilities.squeeze().tolist()
                ):
                    print(f"  {label}: {prob:.3f}")
            print(f"Predicted Emotion: {predicted_emotions}")
            print(" ")

        # Go through all results and find the set of emotions that at least NUM_ANSWERS/2 models predicted
        final_answer = []
        for emotion, votes in voting_table.items():
            if votes >= NUM_ANSWERS / 2:
                final_answer.append(emotion)

        print("Voting table:", voting_table)
        print("\nExpected answer:", answer)

        final_answer_text = "Final answer:"

        if not final_answer:
            print(final_answer_text, "none")
        else:
            print(final_answer_text, final_answer)

        # compare final answer with expected answer
        if set(final_answer) == set(answer):
            print("Correct!")
        else:
            # show how many emotions are correct
            correct_emotions = set(final_answer) & set(answer)
            print(
                "Correct emotions:",
                str(len(correct_emotions) / len(answer) * 100) + "%",
            )


if __name__ == "__main__":
    prompt()
    print(" ")
    print("-" * 50)
    print(" ")
