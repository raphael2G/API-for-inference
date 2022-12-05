from fastapi import FastAPI
import os
from inference.classification_inference import run_classification_inference
import urllib

app = FastAPI()

base_data_path = '../dummy_data/images'
base_segmentation_path = '../data/segmentation'

@app.get("/classification/{img_path}")
async def classification(img_path: str):

    negative_confidence, positive_confidence = run_classification_inference(os.path.join(base_data_path, img_path))
    # .item() necessary to for json convserion. numpy is conversion is not enabled. must be native python dtype
    return {'Confidence Negative': negative_confidence.item(), 
            'Confidence Positive': positive_confidence.item()}

@app.get("/segmentation/{img_path}")
async def segmentation(img_path: str):

    # get img from img_path
    # generate segmentation - pass img to the model

    new_path = os.path.join(base_segmentation_path, img_path)
    with open(new_path, 'x') as f:
        # save segmentation to this file
        print()

    #return path to saved segmentation file
    return {new_path}

@app.get("/append_ct_dataset/{uri}")
async def append_ct_dataset(uri: str):
    # code to add image to dataset
    response = urllib.request.urlopen(uri)
    with open('image.jpg', 'wb') as f:
        f.write(response.file.read())

@app.get("/infection_mask_dataset/{uri}")
async def infection_mask_dataset(uri: str):
    # code to add image to dataset
    response = urllib.request.urlopen(uri)
    with open('image.jpg', 'wb') as f:
        f.write(response.file.read())

