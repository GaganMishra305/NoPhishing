# NoPhishing

ML based complete phishing solution for individuals.

## 1. Training Pipeline
To run the training pipeline, after setting the virtual environment, run
```python
python train/main.py -ep 10 -bs 64 -mr 10000 
# add the flag details below
``` 

## 2. Running the backend
```shell
docker pull 9453585091/nophishing-api:latest
docker run -p 5000:5000 9453585091/nophishing-api:latest
```

## For-Future:
* integrate visual-cnn(PhishPedia) and 
* safe-browsing-API v5