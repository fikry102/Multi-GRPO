from PIL import Image
import torch

from .pickscore_scorer import PickScoreScorer
from transformers import AutoProcessor, GitForCausalLM


def make_ocr_color_rewards(device):
    from .ocr import OcrWithColorScorer

    checkpoint = "microsoft/git-large-vqav2"
    processor = AutoProcessor.from_pretrained(checkpoint)
    vqa = GitForCausalLM.from_pretrained(checkpoint).eval().to(device)
    answer_ids = processor.tokenizer.convert_tokens_to_ids(["yes", "no"])
    ocr = OcrWithColorScorer()
    pick = PickScoreScorer(device=device)

    @torch.no_grad()
    def score(images, prompts, metadata, only_strict=True):
        pixels = (images * 255).round().clamp(0, 255).to(torch.uint8)
        arrays = pixels.permute(0, 2, 3, 1).cpu().numpy()
        ocr_scores, boxes = ocr(arrays, metadata)
        color_scores = []
        originals = [Image.fromarray(array) for array in arrays]
        for image, box, item in zip(originals, boxes, metadata):
            region = image.crop(box) if box and box[2] > box[0] and box[3] > box[1] else image
            pixels = processor(images=region, return_tensors="pt").pixel_values.to(device)
            question = processor.tokenizer(f"{item['color']} text?", add_special_tokens=False).input_ids
            ids = torch.tensor([[processor.tokenizer.cls_token_id, *question]], device=device)
            logits = vqa(pixel_values=pixels, input_ids=ids).logits[0, -1, answer_ids]
            color_scores.append(logits.float().softmax(dim=-1)[0].item())
        return {"ocr_with_color": ocr_scores, "vqa_color": color_scores,
                "pickscore": pick(prompts, originals).detach()}, {}

    return score
