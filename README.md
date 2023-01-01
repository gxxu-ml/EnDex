# EnDex: A toolkit to automatically evaluate text Engagingness.
The official repository for the Findings of EMNLP 2022 Paper, **EnDex: Evaluation of Dialogue Engagingness at Scale**.

A link to paper available from here: [https://arxiv.org/pdf/2210.12362.pdf]


## Release Training Data
See the following link to download [https://drive.google.com/file/d/1Jjpne-1CoOp8Ej5QFeX8ry_2plasc1pc/view?usp=sharing]. 
The files with suffix ns means they include negative sampled data. Our released off-the-shelf model is trained on an 80k split with negative sampled data. For details about negative sampling, please read the paper. 

We provided multiple data splits, so that interested researchers can try to train different on those splits to test performance.



## Off-the-shelf Model.

The model can be manually downloaded from this link: [https://drive.google.com/file/d/1ph4P471n0LoM1vbsarj3wvEkpijhwCms/view?usp=sharing]
It is too large to be successfully downloaded in script. 

Then, unzip the file into a folder named endex_model0
```
unzip endex_model0.zip
```

## Usage


**Please make sure to have appropriate versions of pytorch and huggingface transformers installed **
```
Our versions:
torch.__version__ == '1.10.1+cu111'
transformers.__version__ == '4.14.0'
```

After you have trained/downloaded a model, you can run the following script to run inference on an example text sentence. 0 means non-engaging and 1 means engaging. 

```
import torch
from transformers import RobertaTokenizer,RobertaForSequenceClassification
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
model_dir = 'the/directory/to/your/unzipped_folder'
model = RobertaForSequenceClassification.from_pretrained(model_dir)
inputs = tokenizer("it's such a great point, and i'd love to hear back on your thoughts!", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_class_id = logits.argmax().item()
print('the engagingness prediction of the input sentence is: ', predicted_class_id)
```





