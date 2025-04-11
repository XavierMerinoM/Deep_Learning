from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.methods import SmoothGradCAMpp, CAM, LayerCAM, GradCAM
from torchcam.utils import overlay_mask

# Function to detect the attention regions in the image
def attention_regions(image_path, model):
    # Loading new image
    img = read_image(image_path)
    
    # Preprocess it for your chosen model
    input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    with GradCAM(model) as cam_extractor:
        # Preprocess your data and feed it to the model
        out = model(input_tensor.unsqueeze(0))
        # Retrieve the CAM by passing the class index and the model output
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

    # Resize the CAM and overlay it
    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)

    # Return the image with the regions
    return result