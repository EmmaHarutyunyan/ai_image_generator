from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

def main():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to("cpu")

    user_prompt = input("Enter your prompt: ")

    safe_prompt = f"{user_prompt}, bright, cheerful, colorful, friendly, soft lighting, highly detailed"

    print(" Generating your image...")

    image = pipe(
        prompt=safe_prompt,
        height=512,
        width=512,
        num_inference_steps=30,   
        guidance_scale=7.5        
    ).images[0]

    filename = "generated_image.png"
    image.save(filename)
    image.show()

    print(f"Done! Your image is saved as '{filename}'")

if __name__ == "__main__":
    main()
