from .radsegencoder import RADSegEncoder

class RadSegEmbeddingEngine:

    def _init

    def _init_model


  rgb_img = torchvision.io.read_image(rgb_path)
  rgb_img = rgb_img.float() / 255
  rgb_img = torch.nn.functional.interpolate(
    rgb_img.unsqueeze(0),size=(512, 512))

  labels = ["car", "person"]

  enc = RADSegEncoder(model_version="c-radio_v3-b", lang_model="siglip2")
  
  feat_map = enc.encode_image_to_feat_map(rgb_img)