from main import select_model
import torch 
import numpy as np
import random
import os
from data.dataset import *
from torch.utils.data import Dataset, DataLoader
from misc.misc import *




if __name__=='__main__':
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    print(tokenizer.tokenize("Creepy place here!!!", add_special_tokens=True))
    quit()

    model = select_model("max", "roberta-base", 6, 'cpu')
    checkpoint = torch.load(os.path.join(os.path.dirname(__file__), '../outputs/models/RobertaForSequenceClassificationMaxPooling_fold_3_epoch_8.pth'), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    test_dataset = EmotionData(os.path.join(os.path.dirname(__file__), '../data/eng_a_parsed_test.json'), "roberta-base", "reduced")
    test_loader = DataLoader(test_dataset, batch_size=1)

    stats = {
        'train' : {
            'loss': list(),
            'accuracy' : {
                'macro' : list(),
                'micro' : list(),
            },
            'precision' : {
                'macro' : list(),
                'micro' : list(),
            },
            'recall' : {
                'macro' : list(), 
                'micro' : list(),
            },
            'f1' : {
                'macro' : list(), 
                'micro' : list(),
            }
        },
        'val' : {
            'loss': list(),
            'accuracy' : {
                'macro' : list(),
                'micro' : list(),
            },
            'precision' : {
                'macro' : list(),
                'micro' : list(),
                'weighted' : list()
            },
            'recall' : {
                'macro' : list(), 
                'micro' : list(),
                'weighted' : list(),
            },
            'f1' : {
                'macro' : list(), 
                'micro' : list(),
                'weighted' : list(),
            }
        },
        'test' : {
            'loss': list(),
            'accuracy' : {
                'macro' : list(),
                'micro' : list(),
            },
            'precision' : {
                'macro' : list(),
                'micro' : list(),
                'weighted' : list(),
            },
            'recall' : {
                'macro' : list(), 
                'micro' : list(),
                'weighted' : list(),
            },
            'f1' : {
                'macro' : list(), 
                'micro' : list(),
                'weighted' : list(),
            }
        }
    }

    confusion = init_confusion(6)
    with torch.no_grad():
        print(f'Start Testing on Test Dataset!')
        for batch in test_loader:
            input_ids, attention_mask, labels = batch['input_ids'].to('cpu'), batch['attention_mask'].to('cpu'), batch['labels'].to('cpu')
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # convert output to binary format
            bin_out = torch.where(torch.sigmoid(outputs) >=0.5, torch.tensor(1.0), torch.tensor(0.0))
            # collect confusion statistics:
            for i in range(6):
                for j in range(len(bin_out)):
                    if bin_out[j][i] == 1.0 and labels[j][i] == 1.0:
                        confusion[i]['tp'] += 1
                    elif bin_out[j][i] == 0.0 and labels[j][i] == 0.0:
                        confusion[i]['tn'] += 1
                    elif bin_out[j][i] == 1.0 and labels[j][i] == 0.0:
                        confusion[i]['fp'] += 1
                    elif bin_out[j][i] == 0.0 and labels[j][i] == 1.0:
                        confusion[i]['fn'] += 1

    # calculate confusion stats in macro and micro strategy manner
    # macro strategy
    acc = [accuracy(confusion[i]['tp'], confusion[i]['tn'], confusion[i]['fp'], confusion[i]['fn']) for i in range(6)]
    prec = [precision(confusion[i]['tp'], confusion[i]['fp']) for i in range(6)]
    rec = [recall(confusion[i]['tp'], confusion[i]['fn']) for i in range(6)]
    f1 = [f1_score(confusion[i]['tp'], confusion[i]['fp'], confusion[i]['fn']) for i in range(6)]
    # insert values in dict
    stats['test']['accuracy']['macro'].append(sum(acc)/len(acc))
    stats['test']['precision']['macro'].append(sum(prec)/len(prec))
    stats['test']['recall']['macro'].append(sum(rec)/len(rec))
    stats['test']['f1']['macro'].append(sum(f1)/len(f1))
    # micro strategy
    tps = sum([confusion[i]['tp'] for i in range(6)])
    tns = sum([confusion[i]['tn'] for i in range(6)])
    fps = sum([confusion[i]['fp'] for i in range(6)])
    fns = sum([confusion[i]['fn'] for i in range(6)])
    # insert values in dict
    stats['test']['accuracy']['micro'].append(accuracy(tps, tns, fps, fns))
    stats['test']['precision']['micro'].append(precision(tps, fps))
    stats['test']['recall']['micro'].append(recall(tps, fns))
    stats['test']['f1']['micro'].append(f1_score(tps, fps, fns))   
    # weighted strategy
    support = [(confusion[i]['tp'] + confusion[i]['fn']) for i in range(6)]
    # insert values in dict
    stats['test']['precision']['weighted'].append(sum([precision(confusion[i]['tp'], confusion[i]['fp'])*support[i] for i in range(6)])/sum(support))
    stats['test']['recall']['weighted'].append(sum([recall(confusion[i]['tp'], confusion[i]['fn'])*support[i] for i in range(6)])/sum(support))
    stats['test']['f1']['weighted'].append(sum([f1_score(confusion[i]['tp'], confusion[i]['fp'], confusion[i]['fn'])*support[i] for i in range(6)])/sum(support))

    print(stats)
