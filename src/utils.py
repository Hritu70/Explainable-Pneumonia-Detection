
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

label_map = {
    0: "NORMAL",
    1: "PNEUMONIA"
}

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )
])

def predict_image(model, image, device):

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = F.softmax(outputs, dim=1)

        confidence, prediction = torch.max(probabilities, dim=1)

    prediction = prediction.item()
    confidence = confidence.item()

    return (
        label_map[prediction],
        confidence
    )
