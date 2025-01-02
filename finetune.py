import torch
from transformers import LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.utils.dataset_utils import get_dataloader
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from dataclasses import asdict
from llama_recipes.configs import lora_config as LORA_CONFIG
import torch.optim as optim
from llama_recipes.utils.train_utils import train
from torch.optim.lr_scheduler import StepLR


def load_model():
    train_config = TRAIN_CONFIG()
    train_config.model_name = "meta-llama/Meta-Llama-3.1-8B"
    train_config.num_epochs = 1
    train_config.run_validation = False
    train_config.gradient_accumulation_steps = 4
    train_config.batch_size_training = 1
    train_config.lr = 3e-4
    train_config.use_fast_kernels = True
    train_config.use_fp16 = True
    train_config.context_length = (
        1024 if torch.cuda.get_device_properties(0).total_memory < 16e9 else 2048
    )  # T4 16GB or A10 24GB
    train_config.batching_strategy = "packing"
    train_config.output_dir = "meta-llama-semeval"
    train_config.use_peft = True

    config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    model = LlamaForCausalLM.from_pretrained(
        train_config.model_name,
        device_map="auto",
        quantization_config=config,
        use_cache=False,
        attn_implementation="sdpa" if train_config.use_fast_kernels else None,
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, train_config


def check_base_model(model, tokenizer):
    eval_prompt = """
    Summarize this dialog:
    A: Hi Tom, are you busy tomorrow’s afternoon?
    B: I’m pretty sure I am. What’s up?
    A: Can you go with me to the animal shelter?.
    B: What do you want to do?
    A: I want to get a puppy for my son.
    B: That will make him so happy.
    A: Yeah, we’ve discussed it many times. I think he’s ready now.
    B: That’s good. Raising a dog is a tough issue. Like having a baby ;-)
    A: I'll get him one of those little dogs.
    B: One that won't grow up too big;-)
    A: And eat too much;-))
    B: Do you know which one he would like?
    A: Oh, yes, I took him there last Monday. He showed me one that he really liked.
    B: I bet you had to drag him away.
    A: He wanted to take it home right away ;-).
    B: I wonder what he'll name it.
    A: He said he’d name it after his dead hamster – Lemmy  - he's  a great Motorhead fan :-)))
    ---
    Summary:
    """

    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

    model.eval()
    with torch.inference_mode():
        print(
            tokenizer.decode(
                model.generate(**model_input, max_new_tokens=100)[0],
                skip_special_tokens=True,
            )
        )

    return model_input, model


def load_the_preprocessed_dataset(tokenizer, train_config):
    dataset = ""  # TODO: fill in the dataset path

    train_dataloader = get_dataloader(tokenizer, dataset, train_config)
    eval_dataloader = get_dataloader(tokenizer, dataset, train_config, "val")

    return train_dataloader, eval_dataloader


def prepare_model_for_PEFT(model):
    lora_config = LORA_CONFIG()
    lora_config.r = 8
    lora_config.lora_alpha = 32
    lora_dropout: float = 0.01

    peft_config = LoraConfig(**asdict(lora_config))

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    return model


def fine_tune_the_model(
    model, tokenizer, train_dataloader, eval_dataloader, train_config
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


def save_model_checkpoint(model, train_config):
    model.save_pretrained(train_config.output_dir)


def evaluate_model(model, tokenizer, model_input):
    model.eval()
    with torch.inference_mode():
        print(
            tokenizer.decode(
                model.generate(**model_input, max_new_tokens=100)[0],
                skip_special_tokens=True,
            )
        )


# TODO:
# - angucken, wie das Dataset aussieht, was die in dem Tutorial aussieht, um zu gucken, wie unseres aussiehen sollte
# - Überlegen, wie das Dataset preprocesset werden sollte


def main():
    # Step 1: Load the model
    model, tokenizer, train_config = load_model()
    print("Model loaded")

    # Step 2: Check the base model
    model_input, model = check_base_model(model, tokenizer)
    print("Base model checked")

    # Step 3: Load the preprocessed dataset
    train_dataloader, eval_dataloader = load_the_preprocessed_dataset(
        tokenizer, train_config
    )
    print("Dataset loaded")

    # Step 4: Prepare the model for PEFT
    model = prepare_model_for_PEFT(model)
    print("Model prepared for PEFT")

    # Step 5: Fine-tune the model
    results = fine_tune_the_model(
        model, tokenizer, train_dataloader, eval_dataloader, train_config
    )
    print("Model fine-tuned")

    # Step 6: Save the model checkpoint
    save_model_checkpoint(model, train_config)
    print("Model checkpoint saved")

    # Step 7: Evaluate the model
    evaluate_model(model, tokenizer, model_input)
    print("Model evaluated")


if __name__ == "__main__":
    main()
