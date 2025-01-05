import random
from statistics import mode

import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# Define emotion labels
EMOTION_LABELS = [
    "anger",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "none",
]

NUM_ANSWERS = 5
PROMPT = "This is a very exciting and happy moment!"

USE_TOP_K = True
K = 3  # top-k sampling

USE_TEMPERATURE = False
MIN_TEMPERATURE = 0.2
MAX_TEMPERATURE = 0.8


def main():
    print(" ")
    print("-" * 50)
    print(f"Prompt: {PROMPT}")

    all_results = []

    random.seed(42)

    # Load pre-trained model and tokenizer
    model_name = "roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=len(EMOTION_LABELS), cache_dir="cache-dir"
    )

    # Tokenize the input
    inputs = tokenizer(PROMPT, return_tensors="pt", truncation=True, padding=True)

    for i in range(NUM_ANSWERS):
        print("Run:", i + 1)

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)

        # add a random temperature
        if USE_TEMPERATURE:
            temperature = random.uniform(MIN_TEMPERATURE, MAX_TEMPERATURE)
            logits = outputs.logits / temperature
            print(f"Temperature: {temperature:.2f}")
        else:
            logits = outputs.logits

        # Get predicted probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # add top-k sampling
        if USE_TOP_K:
            top_k_probs, top_k_indices = torch.topk(probabilities, K)
            sampled_index = torch.multinomial(top_k_probs, num_samples=1).item()
            predicted_class = top_k_indices[0][sampled_index].item()
            print("Top-K sampling:")
            for prob, idx in zip(
                top_k_probs.squeeze().tolist(), top_k_indices.squeeze().tolist()
            ):
                print(f"  {EMOTION_LABELS[idx]}: {prob:.3f}")
        else:
            # Get predicted class (emotion)
            predicted_class = torch.argmax(probabilities, dim=-1).item()

        predicted_emotion = EMOTION_LABELS[predicted_class]

        all_results.append(predicted_emotion)

        print("Probabilities:")
        for label, prob in zip(EMOTION_LABELS, probabilities.squeeze().tolist()):
            print(f"  {label}: {prob:.3f}")
        print(f"==> Predicted Emotion: {predicted_emotion}")
        print(" ")

    most_common = mode(all_results)
    print(f"Most common emotion: {most_common}")
    print("-" * 50)
    print(" ")


if __name__ == "__main__":
    main()
