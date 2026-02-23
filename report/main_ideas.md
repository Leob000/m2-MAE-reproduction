# Motivations
- Models now easily overfit over large image datasets -> Demand for big labeled datasets -> Human labeling is a bottleneck -> Inspiration of NLP SSL methods (BERT, GPT) -> Application to vision
- CNN were difficult to mask, recent new ViT architecture are easy to mask (working with patches)
- Information density less in images than text -> Mask a high % of the image
  - **Optimization ↔ Vision:** High masking ratio (75%) shifts the optimization objective from simple low-level pixel interpolation to high-level semantic reconstruction.

# Model
- Autoencoder: Encoder that maps the input to a latent space, and a (temporary) decoder that reconstructs the input from the latent space
- Both the encoder and decoder are "typical" ViT; the input image is split into patches like in typical ViT. Quickly explain the ViT operations/formulas?
- Masks random patches from the input image (usually 75%), reconstruct the missing patches in the pixel space
  - "Random sampling with a high masking ratio eliminates redundancy, the task cannot be easily solved by extrapolation from visible neighboring patches"
- Asymetric encoder/decoder: Encoder operates only on the visible patches, we only give the masked patches to the decoder (reducing compute and memory), decoder is lightweight and discarded after pretraining
  - **Optimization ↔ Vision:** The asymmetric design is a hardware-conscious optimization. By processing only ~25% of patches in the heavy encoder, it allows for training much larger models (scaling) within the same compute budget.
- Encoder: ViT
  - Only on visible patches -> reduces Sequence length -> reduces compute and memory, allowing to scale up the model
  - Gets positional encoding before the masking
- Decoder: ViT
  - Lightweight, smaller than the encoder (no need for it to be big for the pretraining to work; <10% of the computation of the original tokens) and discarded after pretraining; independadnt from the encoder
  - Gets all the patches (encoded visible patches and masked) as input
  - The masked patches are "represented" by learnable mask tokens
  - We add the positional embeddings again
- Reconstruction
  - Pixel space
  - Last layer is a decoder, output dimension is the number of pixels in a patch
  - MSE Loss on masked patches
    - **Optimization ↔ Vision:** MSE in pixel space (especially with per-patch normalization) serves as a simple yet effective proxy for learning visual representations, bypassing the need for complex contrastive pairs or external tokenizers.
  - The reconstruction target is the normalized (per patch) pixel values

# Their results
- Transfer learning: Classification, Object detection, Segmentation, Semantic segmentation (good results + better results by scaling up the models; aligned with the scaling in NLP)
- Quickly talk about their results VS other existing methods at the time (other models; supervised methdos)
- Must talk among other things: Masking ratio, wall-clock time, training schedule (allowing a high number of epochs)

# Our reproduction (experiments, results, ...)
- Quickly explain our code
- Results of experiments from wandb
- Quickly talk about the notebook
- high difference in our results between linear probing and finetuning -> the paper talks about linear probing missing the option of pursuing non linear features
- Have a full table/list of hyperparameters we used to be able to reproduce our results (see GEMINI.md for the hyperparameters we used)
- Compare our results to the results of the paper and the smaller scale paper, try to impute why we have bad linear probing comparaed to finetuning

# Other
Disussion and limitations
What came after the paper
Quick word about other SSL methods? (contrastive learning, others?; maybe in the intro of the report insteaad?)
