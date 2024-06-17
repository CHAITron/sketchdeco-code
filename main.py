import argparse, os
from global_stage import GlobalColourise
from local_stage import LocalColourise
import torch
from diffusers.utils import load_image

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="input directory",
        default="./inputs"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./outputs"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="the seed (for reproducible sampling)",
    )

    parser.add_argument(
        "--tau",
        type=float,
        help="",
        default=0.4
    )

    opt = parser.parse_args()
    # init global and local steps
    global_step = GlobalColourise()
    local_step = LocalColourise(opt.outdir)
    for subdir, _, files in os.walk(opt.indir):

        palettes_path = []
        palettes_list = []
        masks_path = []
        global_results = []

        for file in files:
            torch.cuda.empty_cache()
            file_path = os.path.join(subdir, file)

            if file.endswith('.txt'):
                # process palettes file
                palettes_path.append(file_path)
            else:
                if file.startswith('sketch'):
                    sketch_path = file_path
                elif file.startswith('mask'):
                    masks_path.append(file_path)
                    
            if file == files[-1]:
                masks_path = sorted(masks_path, key=lambda x: int(''.join(filter(str.isdigit, x))))
                palettes_path = sorted(palettes_path, key=lambda x: int(''.join(filter(str.isdigit, x))))
                for palette_path in palettes_path:
                    mask_palettes_list = []
                    with open(palette_path, "r") as file:
                        for palette in file:
                            mask_palettes_list.append(palette.strip())
                    palettes_list.append(mask_palettes_list)
                # load sketch
                sketch = load_image(sketch_path)
                
                # perform global colourisation
                print("start global colourisation process...")
                blip_cls = global_step.blip_predict(sketch)
                for palette in palettes_list:
                    global_result = global_step(sketch, blip_cls, opt.seed, palette)
                    global_results.append(global_result)
                print("generating background image")
                # generate background image
                bg = global_step(sketch, blip_cls, opt.seed, bg=True)
                
                #perform local colourisation
                local_step(blip_cls, masks_path, global_results, bg, opt.tau, opt.seed)

if __name__ == "__main__":
    main()

