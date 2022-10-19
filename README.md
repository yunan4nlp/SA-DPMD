# Speaker-Aware Discourse Parsing on Multi-Party Dialogues

Implementation of the paper [Speaker-Aware Discourse Parsing on Multi-Party Dialogues,  COLING'22](https://aclanthology.org/2022.coling-1.477.pdf)

## Requirements

PyTorch 1.9.1+cu111

Python 3.9

Transformers 4.21.1



## Data

[STAC](https://www.irit.fr/STAC/corpus.html)

[Molweni](https://github.com/HIT-SCIR/Molweni)



## PLM

[SSP-BERT](https://drive.google.com/file/d/1NKojZxGXfIBKULuCrBHDwW1pBXVxG3iS/view?usp=sharing)



## Training 

python -u driver/Train.py  --config_file molweni.cfg