# CS4803 EML Final Project
To run q-diffusion, use the following script:  
`python scripts/txt2img.py --prompt "a puppy wearing a hat" --plms --cond --ptq --weight_bit 8 --quant_mode qdiff --no_grad_ckpt --split --n_samples 5 --resume --quant_act --act_bit 8 --sm_abit 16 --outdir ./assets --cali_ckpt checkpoints/sd_w8a8_ckpt.pth`

## Evaluation
- CLIP Score
- CLIP directional similarity
- FID  
https://huggingface.co/docs/diffusers/en/conceptual/evaluation
