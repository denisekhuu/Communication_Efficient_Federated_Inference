from torchvision import transforms
import io
import base64
def image_to_base64(img_tensor):
    img_pil = transforms.ToPILImage()(img_tensor)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")