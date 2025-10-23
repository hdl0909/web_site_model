import io
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model_def import ClassificationImageModel

PATH = 'ml\model.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


@st.cache_data
def load_model():
    model = ClassificationImageModel().to(device)
    model.load_state_dict(torch.load(PATH, weights_only=False))
    model.eval()
    return model

def load_image():
    uploaded_file = st.file_uploader(
        label="Выберите изображение для распознавания"
    )
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None
    
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    img = transform(img)
    return img.unsqueeze(0)

def print_prediction(img, model):
    outputs = model(img.to(device))
    _, predicted = torch.max(outputs, 1)
    st.write(classes[predicted])

model = load_model()
st.title("Классификация изображений")
img = load_image()
result = st.button("Распознать изображение")
if result:
    preprocess_img = preprocess_image(img)
    st.write("**Результаты распознавания:**")
    print_prediction(preprocess_img, model)
