import tritonclient.http as httpclient
import cv2
import numpy as np
import os
from numpy.linalg import norm


client = httpclient.InferenceServerClient(url="localhost:8000")


#Prerocess image after crop
def recog_preprocess(image, mean=127.5, std=0.0078125):
    """
    Return image after preprocess
    """
    input_height = 112
    input_width = 112
    image = cv2.resize(image, (input_height, input_width))
    if not isinstance(image, np.ndarray):
        print('The input should be the ndarray read by cv2!')
        return None
    height, width, channels = image.shape
    if height != input_height or width != input_width:
        print('Input shape mismatch.\,expected shape: (3, 112, 112)')
        return None
    image = (image.transpose((2, 0, 1)) - mean) / std
    image = image.astype(np.float32)
    
    return image

def inference_on_image(image):
    try:
        image = recog_preprocess(image)
    except Exception as e:
        raise e
    image = np.expand_dims(image, axis=0)

    recognition_input = httpclient.InferInput( #create input 
        "input", image.shape, datatype="FP32"
    )

    recognition_input.set_data_from_numpy(image, binary_data=True)

    recognition_response = client.infer(
        model_name="mobilefacenet", inputs=[recognition_input]
    )

    feature = recognition_response.as_numpy("output")
    feature = np.squeeze(feature)

    return feature

#Create feature of target images to compare 
def create_feat_list(list_person_id, path_to_target):
    feat_list = []
    for i in range(len(list_person_id)):
        path_person = os.path.join(path_to_target, list_person_id[i])
        target_name = os.listdir(path_person)[0]
        path_target = os.path.join(path_person,target_name)
        target = cv2.imread(path_target)
        feat = inference_on_image(target)
        feat_list.append(feat)
    return feat_list

#
def iden(feat_img,feat_list):
    scores = []
    for i in range(len(feat_list)):
        cosine = np.dot(feat_list[i],feat_img)/(norm(feat_list[i])*norm(feat_img))
        scores.append(cosine)
    
    max_indices = np.argmax(np.array(scores))
    if scores[max_indices] > 0.95:
        return max_indices, scores[max_indices]
    return -1, scores[max_indices]

if __name__ == "__main__":
    pass