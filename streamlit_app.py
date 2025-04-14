import streamlit as st
import torch
from torch import nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import plotly.graph_objects as go
import numpy as np

#********************** Page configuration
st.set_page_config(page_title="Image Classifier", layout="centered")

model=None

# ************************** Functions
@st.cache_resource
def load_model():

    class pret_neural_resnet(nn.Module):
        def __init__(self,drop_l1=0.5,drop_l2=0.5):
            super().__init__()
            self.pret_model=models.resnet50(weights="DEFAULT")
            
            #Freeze predefined parameters
            for param in self.pret_model.parameters():
                param.requires_grad=False  

            #Unfreeze layer 4 
            for param in self.pret_model.layer4.parameters():
                param.requires_grad=True  

            n_layers=self.pret_model.fc.in_features#Number of input layers
            self.pret_model.fc=nn.Sequential(
                            #Layer 1
                            nn.Linear(n_layers,256),
                            nn.BatchNorm1d(256),
                            nn.ReLU(),
                            nn.Dropout(p=drop_l1),

                            #Layer 2
                            nn.Linear(256,64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Dropout(p=drop_l2),

                            #Layer 3
                            nn.Linear(64,6)
                            )
            
        def forward(self,X):
            return self.pret_model(X)

    model=pret_neural_resnet(0.3,0.5)
    model.load_state_dict(torch.load('best_model_trial_5.pth'))
    model.eval()
    return model

def predict(image):
    global model
    if not model:
        model = load_model()
    img_tensor = transform(image).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0).numpy()
        _,class_predictd=torch.max(outputs,dim=1)
    return probabilities,class_predictd

# ***************************************************************

#**************************** Definitions

# Define transforms
mean_par=torch.tensor([0.4451, 0.4411, 0.4429],dtype=torch.float32)
std_par=torch.tensor([0.2675, 0.2663, 0.2674],dtype=torch.float32)

transform=transforms.Compose([
    transforms.Resize(size=[224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_par,std=std_par)
])


# Define your class names
class_names = {
    0: 'Front Breakage', 
    1: 'Front Crushed', 
    2: 'Front Normal', 
    3: 'Rear Breakage', 
    4: 'Rear Crushed', 
    5: 'Rear Normal'
    } 

#*************************************************************************

#**************************** Streamlit 


st.title("🧠 Image Classifier")

# Context section
st.markdown("""
            ***
    This project is a neural network-based image classifier trained to recognize a car state and classify it into one of six classess:
                 - Front Normal - Front Crushed - Front Breakage - Rear Normal - Rear Crushed - Rear Breakage 
    - **Input**: Upload an image in PNG, JPG, or JPEG format.
    - **Output**: The predicted class and the probability distribution across all classes.               
    
    - **Disclaimers**: The dataset of the project consisted of 2300 images.
            ***
            """
    )
st.markdown("#### Example Image")
st.image("Examples.png", caption="Some examples", use_container_width =True)


uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

if uploaded_file:

    image = Image.open(uploaded_file).convert('RGB')

    probs,_ = predict(image)
    probs=probs.round(2)*100
    top_class = class_names[np.argmax(probs)]

    #Creating a clean frame
    st.markdown("### Your image:")
    st.image(image, caption='Uploaded Image', width= 400)
  
    col1, col2 = st.columns(2)  # wider right column
    
    #Left column
    with col1:
        top_space, center, bottom_space = st.container(), st.container(), st.container()
        with top_space:
            st.markdown(f"### \n")
        with center:
                st.markdown(
        f"""
        <div style="
            background-color: #42a9bd;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            text-align: center;
        ">
            <h3 style="margin-bottom: 10px;">Prediction: </strong></h3>
             <h3 style="margin-bottom: 10px;"> <strong>{top_class} </strong></h3>
             <h3 style="margin-bottom: 10px;">Confidence: <strong>{max(probs):.0f}% </strong></h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    #right column
    with col2:
        fig = go.Figure(data=[
            go.Bar(x=list(class_names.values()), y=probs, marker_color='indianred')
        ])
        fig.update_layout(title='Classification probabilities', xaxis_title='Class', yaxis_title='Probability [%]')
        st.plotly_chart(fig, use_container_width=True)