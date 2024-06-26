# ✨GLORY: Global Graph-Enhanced Personalized News Recommendations
Code adapted from paper [_Going Beyond Local: Global Graph-Enhanced Personalized News Recommendations_](https://arxiv.org/pdf/2307.06576.pdf) published at RecSys 2023. 

<p align="center">
  <img src="glory.jpg" alt="Glory Model Illustration" width="600" />
  <br>
  Glory Model Illustration
</p>


### Dataset download
All files can be found here: https://docs.google.com/forms/d/e/1FAIpQLSdo6YZ1mVewLmqhsqqOjXTKsSp3OmCMHbMjEpsW0t_j-Hjtbg/viewform
Alternatively they can be downloaded individually:
```shell
wget 'https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/google_bert_base_multilingual_cased.zip'
wget 'https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip'
wget 'https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip'
wget 'https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_large.zip'
wget 'https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/articles_large_only.zip'
```


### Environment
> Python 3.8.10
> pytorch 1.13.1+cu117
```shell
cd GLORY

apt install unzip python3.8-venv python3-pip
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python -m spacy download 'da_core_news_md'
```

```shell
# Run
python3 src/main.py model=GLORY dataset=MINDsmall reprocess=True
```


### Changes to original code
- src/dataload/data_preprocess.py
  - Adapt data preprocessing to correctly load and process the EB-NeRD dataset.
  - Use spacy danish NER model to extract entities

- src/models/component/news_encoder.py
  - Use pre-trained danish bert embeddings instead of glove embeddings

- src/main.py
  - Increse validation frequency

- configs/model/default.yaml
  - Updated to support flair and bert embeddings

- src/models/component/news_encoder.py
  - Utilize bert pretrain model as danish word encoder

- src/test.py
  - Same as main.py but just loads the model (load_checkpoint=True) and validates (tests) its

- src/utils/common.py
  - Load build embeddings dictionary and construct entity embedding matrix

#### Assumes access to training and validation datasets (small)
```shell
# Run - Train
python3 src/main.py model=GLORY dataset=EB-NeRD reprocess=True 
```

```shell
# Run - Load & Validate Model
python3 src/test.py model=GLORY dataset=EB-NeRD reprocess=False 
```

### Bibliography

```shell
@misc{yang2023going,
      title={Going Beyond Local: Global Graph-Enhanced Personalized News Recommendations}, 
      author={Boming Yang and Dairui Liu and Toyotaro Suzumura and Ruihai Dong and Irene Li},
      year={2023},
      publisher ={RecSys},
}
```


