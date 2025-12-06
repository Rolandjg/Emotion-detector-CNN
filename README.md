# Emotion-detector-CNN

### Setup
```bash
git clone https://github.com/Rolandjg/Emotion-detector-CNN
cd Emotion-detector-CNN
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

![surpise grad cam](https://github.com/Rolandjg/Emotion-detector-CNN/blob/main/surprise.png)
![happy grad cam](https://github.com/Rolandjg/Emotion-detector-CNN/blob/main/happy.png)

live_cam.py shows a live feed through your camera, classifying your facial expression in real time. <br />
live_grad_cam.py shows a live feed through your camera, classifying your facial expression and showing each convolution layer in real time.

# Training
To train, follow these steps in the root directory of the repo
```
git clone https://github.com/Emilmrk/preprocessed-facial-emotions-224
python3 train.py
```

# Inference
infer.py contains inferences code <br />
infer(PIL_image) takes in an rgb PIL.Image and returns a class string
