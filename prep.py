from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
import torch, os, pickle
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
transform = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.ToTensor()
])

embeddings = {}
for f in tqdm(os.listdir("celebahq")):
    img = Image.open(os.path.join("celebahq", f)).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    emb = model(x*2-1).detach().cpu().numpy()[0]
    embeddings[f] = emb

pickle.dump(embeddings, open("embeddings.pkl","wb"))