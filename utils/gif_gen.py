from PIL import Image
import glob
import matplotlib.pyplot as plt

def gen_gif(folder, output_file="output.gif"):
    files = glob.glob(f"{folder}/*.png")
    files = sorted(files)

    images = [Image.open(img).convert('RGB') for img in files]

    images[0].save(
        output_file,
        save_all=True,
        append_images=images[1:],
        duration=300,              
        loop=0           
    )