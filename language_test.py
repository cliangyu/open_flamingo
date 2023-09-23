import torch
from PIL import Image
import requests
from open_flamingo.src.factory import create_model_and_transforms

model_args = dict(
    vision_encoder_path='ViT-L-14',
    vision_encoder_pretrained='openai',
    lm_path='anas-awadalla/mpt-1b-redpajama-200b',
    lm_tokenizer_path='anas-awadalla/mpt-1b-redpajama-200b',
    cross_attn_every_n_layers='1',
    checkpoint_path='/data/lychen/code/decoding/openflamingo/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt'
)

device = 'cuda'

model, image_processor, tokenizer = create_model_and_transforms(
            model_args["vision_encoder_path"],
            model_args["vision_encoder_pretrained"],
            model_args["lm_path"],
            model_args["lm_tokenizer_path"],
            cross_attn_every_n_layers=int(model_args["cross_attn_every_n_layers"]),
        )
checkpoint = torch.load(model_args["checkpoint_path"], map_location=device)
if "model_state_dict" in checkpoint:
    checkpoint = checkpoint["model_state_dict"]
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
model.load_state_dict(checkpoint, strict=False)
model.to(device)
model.eval()


"""
Step 1: Load images
"""
demo_image_one = Image.open(
    requests.get(
        "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
    ).raw
)

demo_image_two = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028137.jpg",
        stream=True
    ).raw
)

query_image = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028352.jpg", 
        stream=True
    ).raw
)


"""
Step 2: Preprocessing images
Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
 batch_size x num_media x num_frames x channels x height x width. 
 In this case batch_size = 1, num_media = 3, num_frames = 1,
 channels = 3, height = 224, width = 224.
"""
vision_x = [image_processor(demo_image_one).unsqueeze(0), image_processor(demo_image_two).unsqueeze(0), image_processor(query_image).unsqueeze(0)]
vision_x = torch.cat(vision_x, dim=0)
vision_x = vision_x.unsqueeze(1).unsqueeze(0).to(device)


tokenizer.padding_side = "left" # For generation padding tokens should be on the left
lang_x = tokenizer(
    ["<image>An image of two cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|><image>An image of"],
    return_tensors="pt",
).to(device)

# visual case
output = model.generate(
    vision_x=vision_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=20,
    num_beams=3,
)
generated_text = tokenizer.batch_decode(output)
print(generated_text)

# blind case
model.lang_encoder.clear_conditioned_layers()
output = model.lang_encoder.generate(
    input_ids=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=20,
    num_beams=3,
)
generated_text = tokenizer.batch_decode(output)
print(generated_text)
# '<image>An image of two cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|><image>An image of a bathroom sink with a faucet and a mirror on the wall behind it.\n"The'

# remove the image related tokens
lang_x = tokenizer(
    ["An image of two cats.An image of a bathroom sink.An image of"],
    return_tensors="pt",
).to(device)
# blind case
model.lang_encoder.clear_conditioned_layers()
output = model.lang_encoder.generate(
    input_ids=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=20,
    num_beams=3,
)
generated_text = tokenizer.batch_decode(output)
print(generated_text)
# An image of two cats.An image of a bathroom sink.An image of a bathroom sink.An image of a bathroom sink.An image of a bathroom sink.An image