from transformers import AutoProcessor, BlipForQuestionAnswering
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from scipy.spatial import KDTree
from webcolors import (
    CSS3_HEX_TO_NAMES,
    hex_to_rgb,
)
from PIL import ImageColor, ImageOps

class GlobalColourise:
    def __init__(self):
        print("initialize global colourisation components...")
        # initialize SD1.5+controlnet model
        controlnet = ControlNetModel.from_pretrained(
                                "lllyasviel/sd-controlnet-scribble", 
                                torch_dtype=torch.float16
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                                "runwayml/stable-diffusion-v1-5", 
                                controlnet=controlnet, 
                                safety_checker=None, 
                                torch_dtype=torch.float16
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()

        # initilize blip model
        self.blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

    def __call__(self, sketch, blip_cls, seed, palette=[], bg=False):
        # apply global sketch colourisation
        # 1: preprocess sketch image
        sketch_rz = sketch.resize((512, 512))
        sketch_inv = ImageOps.invert(sketch_rz)
        if not bg:
            print(f"perform global colourisation for palette: {palette}")
            # 2: get colour names, e.g., ["#800080","#DA70D6","#FFD700"]
            rgb_colors = list(map(lambda n: ImageColor.getcolor(n, "RGB"), palette))
            name_colors = list(map(lambda n: self.convert_rgb_to_names(n), rgb_colors))
            # 3: generate result
            colour_str = ','.join(name_colors)
            txt_ppt = "{}, {}, {}".format(
                        blip_cls,
                        "hyper-realistic, quality, photography style",
                        "using only colors in color pallete of {}".format(colour_str),
            )
        else:
            txt_ppt = "{}, {}".format(
                        blip_cls,
                        "hyper-realistic, quality, photography style",
            )
        neg_ppt = (
            'drawing look, sketch look, line art style, cartoon look, ' 
            'unnatural color, unnatural texture, unrealistic look, low-quality'
        )
        print(f"generate result...")
        result = self.pipe(
            prompt=txt_ppt, 
            image=sketch_inv,
            generator=torch.Generator().manual_seed(seed), 
            num_inference_steps=20, 
            negative_prompt=neg_ppt
        )
        print("DONE")

        return result.images[0]

    def blip_predict(self, sketch):
        # predict sketch class from image
        text = "what is the object in this sketch?"
        inputs = self.processor(images=sketch, 
                                text=text, 
                                return_tensors="pt"
        )
        outputs = self.blip_model.generate(**inputs)
        blip_cls = self.processor.decode(outputs[0], skip_special_tokens=True)
        return blip_cls

    @staticmethod
    def convert_rgb_to_names(rgb_tuple):

        # a dictionary of all the hex and their respective names in css3
        css3_db = CSS3_HEX_TO_NAMES
        names = []
        rgb_values = []
        for color_hex, color_name in css3_db.items():
            names.append(color_name)
            rgb_values.append(hex_to_rgb(color_hex))

        kdt_db = KDTree(rgb_values)
        _, index = kdt_db.query(rgb_tuple)
        return names[index]