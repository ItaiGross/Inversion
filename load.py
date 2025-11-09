import pickle

# load dictionary of {filename: embedding_vector}
with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

# access one entry
print(len(embeddings))                # number of samples
print(list(embeddings.keys())[:5])    # filenames
vec = embeddings["00001.jpg"]        # numpy array shape (512,)

# convert to torch tensor when used
import torch
emb_target = torch.tensor(vec).unsqueeze(0).cuda()  # shape [1,512]
print(emb_target.shape)