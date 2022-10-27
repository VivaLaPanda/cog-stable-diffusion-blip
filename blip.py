import clip
import gc
import os
import torch

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from BLIPPY.models.blip import blip_decoder

class ImageDescriber():
    def __init__(
        self
    ):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.blip_image_eval_size = 384
        blip_model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'        
        blip_model = blip_decoder(pretrained=blip_model_url, image_size=self.blip_image_eval_size, vit='base', med_config="./BLIPPY/configs/med_config.json")
        blip_model.eval()
        blip_model = blip_model.to(self.device)
        self.blip_model = blip_model

        data_path = "./BLIPPY/data/"

        self.artists = self.load_list(os.path.join(data_path, 'artists.txt'))
        self.flavors = self.load_list(os.path.join(data_path, 'flavors.txt'))
        self.mediums = self.load_list(os.path.join(data_path, 'mediums.txt'))
        self.movements = self.load_list(os.path.join(data_path, 'movements.txt'))

        sites = ['Artstation', 'behance', 'cg society', 'cgsociety', 'deviantart', 'dribble', 'flickr', 'instagram', 'pexels', 'pinterest', 'pixabay', 'pixiv', 'polycount', 'reddit', 'shutterstock', 'tumblr', 'unsplash', 'zbrush central']
        trending_list = [site for site in sites]
        trending_list.extend(["trending on "+site for site in sites])
        trending_list.extend(["featured on "+site for site in sites])
        trending_list.extend([site+" contest winner" for site in sites])

        self.trending_list = trending_list

    def generate_caption(self, pil_image):
        gpu_image = transforms.Compose([
            transforms.Resize((self.blip_image_eval_size, self.blip_image_eval_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            caption = self.blip_model.generate(gpu_image, sample=False, num_beams=3, max_length=20, min_length=5)
        return caption[0]

    def load_list(self, filename):
        with open(filename, 'r', encoding='utf-8', errors='replace') as f:
            items = [line.strip() for line in f.readlines()]
        return items

    def rank(self, model, image_features, text_array, top_count=1):
        top_count = min(top_count, len(text_array))
        text_tokens = clip.tokenize([text for text in text_array]).cuda()
        with torch.no_grad():
            text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = torch.zeros((1, len(text_array))).to(self.device)
        for i in range(image_features.shape[0]):
            similarity += (100.0 * image_features[i].unsqueeze(0) @ text_features.T).softmax(dim=-1)
        similarity /= image_features.shape[0]

        top_probs, top_labels = similarity.cpu().topk(top_count, dim=-1)  
        return [(text_array[top_labels[0][i].numpy()], (top_probs[0][i].numpy()*100)) for i in range(top_count)]

    def interrogate(self, image, models):
        caption = self.generate_caption(image)
        if len(models) == 0:
            print(f"\n\n{caption}")
            return

        table = []
        bests = [[('',0)]]*5
        for model_name in models:
            print(f"Interrogating with {model_name}...")
            model, preprocess = clip.load(model_name)
            model.cuda().eval()

            images = preprocess(image).unsqueeze(0).cuda()
            with torch.no_grad():
                image_features = model.encode_image(images).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)

            ranks = [
                self.rank(model, image_features, self.mediums),
                self.rank(model, image_features, ["by "+artist for artist in self.artists]),
                self.rank(model, image_features, self.trending_list),
                self.rank(model, image_features, self.movements),
                self.rank(model, image_features, self.flavors, top_count=3)
            ]

            for i in range(len(ranks)):
                confidence_sum = 0
                for ci in range(len(ranks[i])):
                    confidence_sum += ranks[i][ci][1]
                if confidence_sum > sum(bests[i][t][1] for t in range(len(bests[i]))):
                    bests[i] = ranks[i]

            row = [model_name]
            for r in ranks:
                row.append(', '.join([f"{x[0]} ({x[1]:0.1f}%)" for x in r]))

            table.append(row)

            del model
            gc.collect()
        #display(pd.DataFrame(table, columns=["Model", "Medium", "Artist", "Trending", "Movement", "Flavors"]))

        flaves = ', '.join([f"{x[0]}" for x in bests[4]])
        medium = bests[0][0][0]
        if caption.startswith(medium):
            out = f"{caption} {bests[1][0][0]}, {bests[2][0][0]}, {bests[3][0][0]}, {flaves}"
            print(f"\n\n{caption} {bests[1][0][0]}, {bests[2][0][0]}, {bests[3][0][0]}, {flaves}")
        else:
            out = f"{caption}, {medium} {bests[1][0][0]}, {bests[2][0][0]}, {bests[3][0][0]}, {flaves}"
            print(f"\n\n{caption}, {medium} {bests[1][0][0]}, {bests[2][0][0]}, {bests[3][0][0]}, {flaves}")
        return caption, out

    # blip_prompt, clip_inter_prompt = interrogate(image, models=["ViT-L/14"])