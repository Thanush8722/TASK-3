from flask import Flask, render_template, request
import torch
from torchvision import transforms, models
from PIL import Image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Define model structure (must match train_model.py)
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

# ✅ Load trained weights and class mapping
ckpt = torch.load('models/resnet_cat_dog.pth', map_location='cpu')
model.load_state_dict(ckpt["model_state"])
class_to_idx = ckpt["class_to_idx"]
idx_to_class = {v: k for k, v in class_to_idx.items()}

model.eval()

# ✅ Same transform as used in training (with Normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_filename = None

    if request.method == 'POST':
        file = request.files['image']
        if file and file.filename != '':
            fname = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            file.save(image_path)
            image_filename = fname

            image = Image.open(image_path).convert('RGB')
            tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(tensor)
                idx = int(output.argmax(1).item())
                prediction = idx_to_class[idx].capitalize()

    return render_template('index.html',
                           prediction=prediction,
                           image_filename=image_filename)

if __name__ == '__main__':
    app.run(debug=True)
