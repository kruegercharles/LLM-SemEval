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
]

NUM_ANSWERS = 5
PROMPT = "This is a very exciting and happy moment!"


def main():
    # Load pre-trained model and tokenizer
    model_name = "roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=len(EMOTION_LABELS)
    )

    # Tokenize the input
    inputs = tokenizer(PROMPT, return_tensors="pt", truncation=True, padding=True)

    print(" ")
    print("-" * 50)
    print(f"Prompt: {PROMPT}")

    all_results = []

    for i in range(NUM_ANSWERS + 1):
        print("Run:", i + 1)
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Get predicted probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Get predicted class (emotion)
        predicted_class = torch.argmax(probabilities, dim=-1).item()

        predicted_emotion = EMOTION_LABELS[predicted_class]

        all_results.append(predicted_emotion)

        print(f"Predicted Emotion: {predicted_emotion}")
        print("Probabilities:")
        for label, prob in zip(EMOTION_LABELS, probabilities.squeeze().tolist()):
            print(f"{label}: {prob:.3f}")
        print(" ")

    most_common = mode(all_results)
    print(f"Most common emotion: {most_common}")
    print("-" * 50)
    print(" ")


if __name__ == "__main__":
    main()
