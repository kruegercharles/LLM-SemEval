import pandas as pd
import torch
print("Imported: Torch")
import shap
print("Imported: Shap")
from transformers import RobertaForSequenceClassification, RobertaTokenizer
print("Imported: Roberta")
import models
print("Imported: Own Models src")
#from models.lm_classifier import RobertaForSequenceClassificationPure
from transformers import Pipeline
from transformers.modeling_outputs import SequenceClassifierOutput
print("Imported: Transformers")
import json
import pickle

print("Finished Imports")


class CustomTextClassificationPipeline(Pipeline):
    def _sanitize_parameters(self, return_all_scores=None, **kwargs):
        preprocess_params, forward_params, postprocess_params = {}, {}, {}
        if return_all_scores is not None:
            postprocess_params["return_all_scores"] = return_all_scores
        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, inputs):
        return self.tokenizer(inputs, truncation=True, padding=True, return_tensors="pt", return_token_type_ids=False)

    def _forward(self, model_inputs):
        logits = self.model(**model_inputs)
        return SequenceClassifierOutput(logits=logits)

    def postprocess(self, model_outputs, return_all_scores=False):
        logits = model_outputs.logits
        if return_all_scores:
            probs = logits.softmax(dim=-1).tolist()
            return [{"label": self.model.config.id2label[i], "score": score} for i, score in enumerate(probs[0])]
        else:
            probs = logits.softmax(dim=-1)
            scores = probs.max(dim=-1)
            return {"label": self.model.config.id2label[scores.indices.item()], "score": scores.values.item()}

def select_model(name, backbone, num_labels, device):
    if name == 'base':
        return RobertaForSequenceClassificationPure(backbone, num_labels).to(device=device)
    elif name == 'deep':
        return RobertaForSequenceClassificationDeep(backbone, num_labels).to(device=device)
    elif name == 'mean':
        return RobertaForSequenceClassificationMeanPooling(backbone, num_labels).to(device=device)
    elif name == 'max':
        return RobertaForSequenceClassificationMaxPooling(backbone, num_labels).to(device=device)
    elif name == 'attention':
        return RobertaForSequenceClassificationAttentionPooling(backbone, num_labels).to(device=device)
    else:
        raise ValueError('Specified model name is not available!')

testing_flag = False
pth_paths = [("src/MeanPooling.pth","mean", 6)]
# pth_paths = [("src/MeanPooling.pth","mean", 6), ("src/MaxPooling.pth","max", 6)]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data
data_json = None
# data/data_15k.json" or 
with open("data/eng_a_parsed.json", 'r') as file:
    data_json = json.load(file)

data = pd.DataFrame(data_json)

print("Dataframe init")
# Load the model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base", max_len=512)

for pth_path_tuple in pth_paths:
    pth_path, arch_name, output_size = pth_path_tuple
    model = select_model(arch_name, "roberta-base", output_size, device)
    loaded_pth = torch.load(pth_path)
    model.load_state_dict(loaded_pth['model_state_dict'])
    model.device = device
    model.can_generate = model.backbone.can_generate

    # Use Custom Pipeline
    pred = CustomTextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=0
    )

    print("Explain Model: ", pth_path)
    explainer = shap.Explainer(pred)

    # Explain the model
    if testing_flag:
        shap_values = explainer(list(data["sentence"][0:2]))
    else:
        shap_values = explainer(list(data["sentence"]))

    # Dump Pickle
    pickle.dump(shap_values, open(f"shap_{arch_name}_values_fullData.pkl", "wb"))

print("Length of shap_values: ", len(shap_values))
print("Done")