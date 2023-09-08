from annoy import AnnoyIndex
import torch, numpy as np, os, clip, pickle as pkl
from PIL import Image

if __name__ == "__main__":
    
    # ========================= HARD CODED IMAGES PATHS =========================
    BASE_DIR = "/home/ubuntu/fashion_data"
    BASE_IMG_DIR = "/home/ubuntu/fashion_data/fashion_images"
    # ========================= HARD CODED IMAGES PATHS =========================
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)


    pths = []
    for img_nm in os.listdir(BASE_IMG_DIR):
        pths.append(os.path.join(BASE_IMG_DIR, img_nm))

    idx_to_relative_pth = {}

    with torch.no_grad():

        f = 768  # Length of item vector that will be indexed

        t = AnnoyIndex(f, "dot")
        for i, img_pth in enumerate(pths):
            idx_to_relative_pth[i] = img_pth

            image = preprocess(Image.open(img_pth)).unsqueeze(0).to(device)
            image_features = model.encode_image(image)
            t.add_item(i, image_features[0])

            if i % 100 == 0:
                print(i)

        t.build(100) # 100 trees
        t.save(os.path.join(BASE_DIR, "fashion_example_6_4_2023.ann"))

        with open(os.path.join(BASE_DIR, "idx_to_pth_fashion_example_6_4_2023.pkl"), "wb") as f:
            pkl.dump(idx_to_relative_pth, f)
