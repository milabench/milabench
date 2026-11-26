"""Prompts shared by the image generation benchmarks (ComfyUI workflows, vLLM-Omni): generation i of every model
uses IMAGE_PROMPTS[i % len] with seed generation_seed(i), so the first images each benchmark keeps are the same
prompts, in the same order, across models and engines, and their outputs compare side by side."""

IMAGE_PROMPTS = [
    "a red fox in a snowy forest at golden hour, photorealistic",
    "a lighthouse on a rocky coast during a storm, dramatic clouds, oil painting",
    "a bowl of ramen on a wooden table, steam rising, close-up food photography",
    "an astronaut riding a horse across a desert at sunset, cinematic",
    "a cozy reading nook with a cat sleeping on a pile of books, soft window light",
    "a futuristic city street at night with neon signs reflected in puddles",
    "a hummingbird drinking from a red flower, macro photograph, shallow depth of field",
    "a mountain lake at sunrise with mist over the water, wide angle landscape",
]


def image_prompt(index: int) -> str:
    return IMAGE_PROMPTS[index % len(IMAGE_PROMPTS)]


def generation_seed(index: int) -> int:
    """A fixed seed per generation index: the same prompt gets the same seed in every model."""
    return 1000 + index
