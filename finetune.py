import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from llama_recipes.configs import train_config as TRAIN_CONFIG
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from dataclasses import asdict
from llama_recipes.configs import lora_config as LORA_CONFIG
import torch.optim as optim
from llama_recipes.utils.train_utils import train
from torch.optim.lr_scheduler import StepLR
import os
import json
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding


# Config
MODEL_PATH = "models/checkpoints/Llama3.1-8B-Instruct/"
OUTPUT_DIR = "meta-llama-semeval"


def load_model():
    # make sure the file exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found")

    # See: https://github.com/meta-llama/llama-recipes/tree/main/recipes/quickstart/finetuning
    train_config = TRAIN_CONFIG()
    train_config.model_name = MODEL_PATH
    train_config.num_epochs = 1
    train_config.run_validation = False
    train_config.gradient_accumulation_steps = 4
    train_config.batch_size_training = 1
    train_config.lr = 3e-4
    train_config.use_fast_kernels = False
    train_config.use_fp16 = True
    train_config.context_length = (
        1024 if torch.cuda.get_device_properties(0).total_memory < 16e9 else 2048
    )
    train_config.batching_strategy = "packing"
    train_config.output_dir = OUTPUT_DIR
    train_config.use_peft = True

    # FIXME: irgendwie funktioniert es nicht die Quantisierung zu aktivieren
    #    config = BitsAndBytesConfig(
    #        load_in_8bit=True,
    #    )

    model = LlamaForCausalLM.from_pretrained(
        train_config.model_name,
        device_map="auto",
        # quantization_config=config,
        use_cache=False,
        attn_implementation="sdpa" if train_config.use_fast_kernels else None,
        torch_dtype=torch.float16,
        local_files_only=True,
    )

    tokenizer: LlamaTokenizerFast = AutoTokenizer.from_pretrained(
        train_config.model_name
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, train_config


def check_base_model(model: LlamaForCausalLM, tokenizer: LlamaTokenizerFast):
    system_prompt = "There are 5 different categories for emotions: Anger, Fear, Joy, Sadness, Surprise. Classify the following sentence in none or one or more of these categories. Only answer with none or the appropriate category or categories."

    eval_prompt = (
        system_prompt
        + " Sentence: We ordered some food at Mcdonalds instead of buying food at the theatre because of the ridiculous prices the theatre has."
    )

    model_input: BatchEncoding = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

    model.eval()
    with torch.inference_mode():
        print(
            tokenizer.decode(
                model.generate(**model_input, max_new_tokens=50)[0],
                skip_special_tokens=True,
            )
        )

    return model_input, model


def load_the_preprocessed_dataset(
    tokenizer: LlamaTokenizerFast, train_config: TRAIN_CONFIG
):
    dataset_path = "data/codabench_data/train/eng_a_parsed.json"

    # make sure the file exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file {dataset_path} not found")

    with open(dataset_path, "r") as f:
        data = json.load(f)

    dataset: Dataset = Dataset.from_list(data)

    #    tokenized_dataset = DatasetDict.map(
    #        lambda x: tokenizer(x["sentence"], truncation=True, padding="max_length"),
    #        batched=True,
    #    )

    # Split the dataset into training and evaluation sets
    """   train_test_split: DatasetDict = dataset.train_test_split(test_size=0.1)
    train_dataset: Dataset = train_test_split["train"]
    eval_dataset: Dataset = train_test_split["test"]

    assert isinstance(train_dataset, Dataset)
    assert isinstance(eval_dataset, Dataset)

    train_dataloader:DataLoader = DataLoader()
    eval_dataloader:DataLoader = DataLoader()
    """

    # Split the dataset into training and evaluation sets
    train_test_split: DatasetDict = dataset.train_test_split(test_size=0.1)
    train_dataset: Dataset = train_test_split["train"]
    eval_dataset: Dataset = train_test_split["test"]

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True)

    tokenized_train_dataset: Dataset = train_dataset.map(
        tokenize_function, batched=True
    )
    tokenized_eval_dataset: Dataset = eval_dataset.map(tokenize_function, batched=True)

    tokenized_train_dataset.set_format("torch")
    tokenized_eval_dataset.set_format("torch")

    print("Tokenized train dataset:", tokenized_train_dataset)
    print("Tokenized eval dataset:", tokenized_eval_dataset)

    # print first 10 elements of each dataset
    print(
        "First 10 elements of tokenized train dataset:",
        tokenized_train_dataset.to_list[:10],
    )
    print(
        "First 10 elements of tokenized eval dataset:",
        tokenized_eval_dataset.to_list[:10],
    )

    train_dataloader = DataLoader(
        tokenized_train_dataset,
        batch_size=train_config.batch_size_training,
        shuffle=True,
    )
    eval_dataloader = DataLoader(
        tokenized_eval_dataset,
        batch_size=train_config.batch_size_training,
        shuffle=False,
    )
    return train_dataloader, eval_dataloader


""" def create_dataloader(dataset_path, tokenizer, batch_size=4):

    try:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file {dataset_path} not found")

        with open(dataset_path, "r") as f:
            data = json.load(f)

        dataset: Dataset = Dataset.from_list(data)

        def tokenize_function(examples):
            # Format the input for instruction fine-tuning. A simple prompt is used here.
            prompts = [f"Classify the emotions in this sentence: {sentence}" for sentence in examples["sentence"]]

            tokenized_inputs = tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt")

            # Create labels. Assuming multi-label classification.
            labels = []
            for emotion_list in examples["emotions"]:
                label_vector = [0] * len(EMOTION_LABELS) # Initialize with zeros
                for emotion in emotion_list:
                   if emotion in EMOTION_LABELS:
                        label_vector[EMOTION_LABELS.index(emotion)] = 1
                labels.append(label_vector)

            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        EMOTION_LABELS = sorted(list(set(emotion for item in data for emotion in item["emotions"]))) #Extract unique emotion labels
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

        tokenized_dataset.set_format("torch")

        dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


 """


def prepare_model_for_PEFT(model: LlamaForCausalLM):
    lora_config = LORA_CONFIG()
    lora_config.r = 8
    lora_config.lora_alpha = 32
    lora_dropout: float = 0.01

    peft_config = LoraConfig(**asdict(lora_config))

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    return model


def fine_tune_the_model(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizerFast,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    train_config: TRAIN_CONFIG,
):
    model.train()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        None,
        None,
        None,
        wandb_run=None,
    )
    return results


def save_model_checkpoint(model: LlamaForCausalLM, train_config: TRAIN_CONFIG):
    model.save_pretrained(train_config.output_dir)


def evaluate_model(model: LlamaForCausalLM, tokenizer, model_input: BatchEncoding):
    model.eval()
    with torch.inference_mode():
        print(
            tokenizer.decode(
                model.generate(**model_input, max_new_tokens=100)[0],
                skip_special_tokens=True,
            )
        )


def print_checkpoint(message: str):
    # just a helper function to make the output more readable when reaching a checkpoint
    print(" ")
    print("=" * 50)
    print("Checkpoint", message)
    print("=" * 50)
    print(" ")


def main():
    # Step 1: Load the model
    model, tokenizer, train_config = load_model()
    print_checkpoint("Model loaded")

    # Step 2: Check the base model
    model_input, model = check_base_model(model, tokenizer)
    print_checkpoint("Base model checked")

    # Step 3: Load the preprocessed dataset
    train_dataloader, eval_dataloader = load_the_preprocessed_dataset(
        tokenizer, train_config
    )
    print_checkpoint("Dataset loaded")

    exit()

    # Step 4: Prepare the model for PEFT
    model = prepare_model_for_PEFT(model)
    print_checkpoint("Model prepared for PEFT")

    # Step 5: Fine-tune the model
    results = fine_tune_the_model(
        model, tokenizer, train_dataloader, eval_dataloader, train_config
    )
    print("Results of fine-tuning: ", results)
    print_checkpoint("Model fine-tuned")

    # Step 6: Save the model checkpoint
    save_model_checkpoint(model, train_config)
    print_checkpoint("Model checkpoint saved")

    # Step 7: Evaluate the model
    evaluate_model(model, tokenizer, model_input)
    print_checkpoint("Model evaluated")


if __name__ == "__main__":
    main()
