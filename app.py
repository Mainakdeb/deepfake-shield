import torch
import torchvision.transforms as transforms
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

import albumentations
import PIL
from PIL import Image
from blazeface import FaceExtractor, BlazeFace
from architectures import fornet,weights
from isplutils import utils
from architectures import fornet,weights


class DeepfakeDetectorPipeline():
    def __init__(self, device, model_name, model_checkpoint, face_detector_checkpoint, face_detector_anchors):

        self.device = device 
        self.face_policy = 'scale'
        self.face_size = 224
        
        self.net_model = model_name
        self.net = getattr(fornet,self.net_model)().eval().to(device)
        self.net.load_state_dict(torch.load(model_checkpoint))

        self.transforms = utils.get_transformer(self.face_policy, self.face_size, self.net.get_normalizer(), train=False)

        facedet = BlazeFace().to(device)
        facedet.load_weights(face_detector_checkpoint)
        facedet.load_anchors(face_detector_anchors)
        self.face_extractor = FaceExtractor(facedet=facedet)

    def predict_from_image_tensor(self, image_tensor):
        with torch.no_grad():
            pred = torch.sigmoid(self.net(image_tensor.to(self.device))).cpu().numpy().flatten()[0]
        return pred 

    def get_pil_from_nchw(self, nchw_tensor):
        pil = transforms.ToPILImage()(nchw_tensor)
        return pil 

    def predict(self, image, num_faces = 1):
        im_faces_all = self.face_extractor.process_image(img=image)

        pred_scores = []
        detected_faces = []

        for i in range(len(im_faces_all['faces'])):

            im_face_np = im_faces_all['faces'][i]
            
            face_tensor = self.transforms(image=im_face_np)['image'].unsqueeze(0) 
            pred = self.predict_from_image_tensor(image_tensor = face_tensor)

            pred_scores.append(pred)
            detected_faces.append(Image.fromarray(im_face_np))

        D = {
            'scores': pred_scores,
            'faces': detected_faces
        }
        return D

loaded_pipeline = DeepfakeDetectorPipeline(
                    model_name = 'EfficientNetAutoAttB4',
                    model_checkpoint = 'predictor/models/EfficientNetAutoAttB4_DFDC_bestval.pth',
                    face_detector_checkpoint = 'predictor/models/blazeface.pth',
                    face_detector_anchors = 'predictor/models/anchors.npy',
                    device = 'cpu'
                )

class SingleResult():
    def __init__(self, image, score):
        self.image = image 
        self.result = 'fake: '+ str(score*100)[:5] + '%' 
        self.filename = '__temp__.jpg'
        self.fontsize  = 37
        self.height , self.width = 300, 680

    def get_image(self):
        fig, ax =  plt.subplots(1,2, figsize = (10,5))
                
        ax[0].imshow(self.image)
        ax[1].text(
            0.5, 
            0.5, 
            self.result, 
            horizontalalignment='center',
            verticalalignment='center', 
            fontsize = self.fontsize
        )

        ax[0].axis('off')
        ax[1].axis('off')

        fig.tight_layout()
        fig.savefig(self.filename, bbox_inches = 'tight', pad_inches = 0)
        img = Image.open(self.filename).resize((self.width, self.height))

        return img

def concat_pil_images(img_list: list):
    imgs_comb = np.vstack(tuple(img_list))
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    return imgs_comb

def get_image_collage(images, scores, label_font_size=8):

    all_results = [SingleResult(images[i], scores[i]).get_image() for i in range(len(images))]

    final_concat= concat_pil_images(all_results)
    return final_concat

def predict_func_gradio(input_image):
    s = loaded_pipeline.predict( image=input_image)
    num_images = len(s['faces'])
    
    ret=get_image_collage(images=s['faces'], 
                          scores=s['scores'])
    return(ret)

ARTICLE = '''
# How it works
Deep shield helps you detect deepfakes in an image.
'''
examples = [
    ['fake_2.jpg'],
    ['donald_bean_2.jpg'],

]

iface = gr.Interface(
    predict_func_gradio, 
    gr.inputs.Image(image_mode="RGB"),
    [gr.outputs.Image(label="Results")],
    title="Deep-Shield",
    layout="horizontal",
    examples = examples,
    allow_flagging=False,
    article = ARTICLE
)

iface.launch(debug=True)
