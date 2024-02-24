__all__ = ['is_write_off','learn','classify_image','categories','image','lable','examples','intf']

from fastai.vision.all import *
import gradio as gr

def is_write_off(x): return x[0].isupper()
learn = load_learner('model.pkl')

categories = ('Just dented or scratched','Write-off')
def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

image = gr.Image()
label = gr.Label()

examples= ['test.jpg','e2.jpg']

demo = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
demo.launch()
