from typing import List, Optional
import fire
from llama import Llama, Dialog
from statistics import mode


# Config

# Pick one:
USE_CHAT = True
USE_TEXT_GEN = not USE_CHAT

# Llm config:
TEMPERATURE = 0.6 # Default: 0.6
TOP_P = 0.9 # Default: 0.9
MAX_SEQ_LEN = 512 # Default: 256 for text_gen, 512 for chat
MAX_GEN_LEN = 64 # Default: 64
MAX_BATCH_SIZE = 4 # Default: 4

# What sentence to classify
SENTENCE = "About 2 weeks ago I thought I pulled a muscle in my calf"

# We can query the model multiple times and then take the most common answer
NUM_ANSWERS = 5


def text_gen(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature= TEMPERATURE,
    top_p = TOP_P,
    max_seq_len = MAX_SEQ_LEN,
    max_gen_len = MAX_GEN_LEN,
    max_batch_size = MAX_BATCH_SIZE,
):
    """
    Examples to run with the pre-trained models (no fine-tuning). Prompts are
    usually in the form of an incomplete text prefix that the model can then try to complete.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.
    `max_gen_len` is needed because pre-trained models usually do not stop completions naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompts: List[str] = ["There are 5 different categories for emotions: Anger, Fear, Joy, Sadness, Surprise. Classify the following sentence in none or one or more of these categories. Only answer with none or the appropriate category or categories. Sentence: " + SENTENCE]

    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        echo=False,
    )
    for prompt, result in zip(prompts, results):
        print("\n==================================\n")
        print("Prompt:")
        print(prompt)
        print("Result:")
        print(result['generation'])
        print("\n==================================\n")


def chat(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature =TEMPERATURE,
    top_p = TOP_P,
    max_seq_len = MAX_SEQ_LEN,
    max_batch_size = MAX_BATCH_SIZE,
    max_gen_len: Optional[int] = None,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs: List[Dialog] = [
        [
            {"role": "system", "content": "There are 5 different categories for emotions: Anger, Fear, Joy, Sadness, Surprise. Classify the following sentence in none or one or more of these categories. Only answer with none or the appropriate category or categories."},
            {"role": "user", "content": "Sentence: " + SENTENCE},
        ],
    ]

    all_results:List[str] = []

    for i in range(NUM_ANSWERS):
        results = generator.chat_completion(
            dialogs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        all_results.append(str(results[0]['generation']['content']))

    most_common = mode(all_results)


    for dialog, result in zip(dialogs, results):
        print("\n==================================\n")
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")

        print("All answers:")
        for i in range(NUM_ANSWERS):
            print(f"Assistant: {all_results[i]}")

        print("\nMost common answer:")
        print(
            f"Assistant: {most_common}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    if USE_TEXT_GEN:
        fire.Fire(text_gen)
    if USE_CHAT:
        fire.Fire(chat)
