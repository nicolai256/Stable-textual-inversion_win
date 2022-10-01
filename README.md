# Stable Textual Inversion_win

Credits for the original script go to https://github.com/rinongal/textual_inversion, my repo is another implementation.

Please read this tutorial to gain some knowledge on how it works https://www.reddit.com/r/StableDiffusion/comments/wvzr7s/tutorial_fine_tuning_stable_diffusion_using_only/

# Changelog

- [x] Added support for windows
- [x] Added support for img2img + textual inversion
- [x] Added colab notebook that works on free colab for training textual inversion 
- [x] Made fork stable-diffusion-dream repo to support textual inversion etc.
- [X] Fixed saving last.ckpt and embeddings.pt every 500 steps
- [X] Fixed merge_embeddings.pt
- [X] Fixed resuming training
- [X] Added squarize outpainting images

# Setup

Start with installing stable diffusion dependencies

```py
conda env create -f environment.yaml
conda activate ldm
```

**You need to install a couple extra packages on top of the ldm env for this to work**

```py
pip install setuptools==59.5.0
pip install pillow==9.0.1
pip install torchmetrics==0.6.0
pip install -e .
```

# Important info

- **Out of memory error:** Try adding ```--base configs/stable-diffusion/v1-finetune_lowmemory.yaml```
- You can follow the progress of your training by looking at the images in this folder `logs/datasetname` model time `projectname/images`.
- It trains forever until you stop it, so just stop the training whenever you're happy with the result images in `logs/randomname/images`. You can resume training later.

# Training

**WARNING: Under 11/12gb vram gpu's training will not work *(for now atleast)*, but you can use the colab notebook (you'll see it when u scroll down).**

```py
python main.py \
 --base configs/stable-diffusion/v1-finetune.yaml \
 -t --no-test \
 --actual_resume "SD/checkpoint/path" \
 --gpus 0,  \
 --data_root "C:\path\to\images" \
 --init_word "keyword" \
 -n "projectname" \
```

To ask for a init word and project name when running the command, you can create a `.bat` file like this example:

```py
@echo  off
call %UserProfile%\miniconda3\Scripts\activate.bat ldm
set /p "init=Enter init word: "
set /p "projectname=Enter name of object/style: "

python main.py --base configs/stable-diffusion/v1-finetune.yaml -t --no-test --actual_resume "SD/checkpoint/sd-v1-4.ckpt" --gpus 0, --data_root "PATHTOTRAINIMGIMAGES/imgs" --init_word "%init%" -n "%projectname%"
```

**NOTE:** PowerShell, or some terminals may prefer `/` instead of `\` in file paths.

# Stopping / Pausing
**To stop training** use `Ctrl+C`. It will create a checkpoint, and exit.

**To pause training you can double click inside your command prompt (usually):** Selecting text usually pauses command prompt windows, but may not work in some terminals, like VSCode.

- For small datasets 3000-7000 steps are enough. All of this depends depends on the size of the dataset. Check in the images folder to see if you're getting a good result.
- Results of the resumed checkpoint will be saved in the original checkpoint path.
- You can later resume training from a checkpoint...

# Resuming

Make sure your path is specified with forward slashes `/`, instead of back slashes `\`. For example: ```path\path\path``` becomes ```path/path/path``` when resuming.

Check your `./logs` folder to see the last training sessions folder. It is in the format `imgs<DateTime>_<ProjectName>`. Replace `FOLDERNAME` with your folder's name.

```py
python "main.py" \
 --base "configs/stable-diffusion/v1-finetune.yaml" \
 -t --no-test \
 --actual_resume " models/ldm/stable-diffusion-v1/model.ckpt" \
 --gpus 0 \
 --data_root "C:/path/to/training/images" \
 --resume "logs/FOLDERNAME" \
 -l logs \
 --embedding_manager_ckpt "logs/FOLDERNAME/checkpoints/embeddings.pt" \
 --resume_from_checkpoint "logs/FOLDERNAME/checkpoints/last.ckpt"
```

You can use replace `FOLDERNAME` with `%text%`, a variable, and add a prompt when launching. You can type in text when you run the script, instead of needing to modify it. Simply use a `.bat` file, like the example below.

```py
@echo  off
call %UserProfile%\miniconda3\Scripts\activate.bat ldm
set /p "text=Enter last (/logs) folder name: "

python main.py --base configs/stable-diffusion/v1-finetune.yaml -t --actual_resume "SD/checkpoint/sd-v1-4.ckpt" --gpus 0, --data_root "PATHTOTRAINIMGIMAGES/imgs" --resume "logs/%text%" -l logs --embedding_manager_ckpt "logs/%text%/checkpoints/embeddings.pt" --resume_from_checkpoint "logs/%text%/checkpoints/last.ckpt"
```

# Merge trained models together

Make sure you use different symbols in placeholder_strings: ["*"] (in the .yaml file while training) if u want to use this.

```py
python merge_embeddings.py --manager_ckpts "/path/to/first/embedding.pt" "/path/to/second/embedding.pt" [...] --output_path "/path/to/output/embedding.pt"
```

# Running this script elsewhere

## Google Colab
 
 You can use a Google Colab notebook for training if your GPU is not good enough to run this program. The free version of Google Colab is a good place to start. A note on Google Colab is that there are now limits, implemented in a recent update.

Textual inversion/finetuning script: https://colab.research.google.com/drive/1MggyUS5BWyNdoXpzGkroKgVoKlqJm7vI?usp=sharing

## Runpod
You can run this repo on Runpod. See "Running Textual Inversion on RunPod using JuptyrLab": https://github.com/GamerUntouch/textual_inversion

# Generating images
**For easy image generation: Try [lstein/Stable Diffusion](https://github.com/lstein/stable-diffusion).**

Text weights, txt2img, img2img, and Textual Inversion are all supported.

Windows
```py
python ./scripts/dream.py --embedding_path "/path/to/embedding.pt" --full_precision
```
Linux
```py
python3 ./scripts/dream.py --embedding_path "/path/to/embedding.pt" --full_precision
```

---

# Read more about Textual Inversion

To learn more about textual inversion, see the [Official Github.io page](https://textual-inversion.github.io/).

The official repo contains the official code, data and sample inversions for the Textual Inversion paper. Find the [Official GitHub repo](https://github.com/rinongal/textual_inversion).

[![arXiv](https://img.shields.io/badge/arXiv-2208.01618-b31b1b.svg)](https://arxiv.org/abs/2208.01618)

> **An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion**<br>
> Rinon Gal<sup>1,2</sup>, Yuval Alaluf<sup>1</sup>, Yuval Atzmon<sup>2</sup>, Or Patashnik<sup>1</sup>, Amit H. Bermano<sup>1</sup>, Gal Chechik<sup>2</sup>, Daniel Cohen-Or<sup>1</sup> <br>
> <sup>1</sup>Tel Aviv University, <sup>2</sup>NVIDIA

>**Abstract**: <br>
> Text-to-image models offer unprecedented freedom to guide creation through natural language.
  Yet, it is unclear how such freedom can be exercised to generate images of specific unique concepts, modify their appearance, or compose them in new roles and novel scenes.
  In other words, we ask: how can we use language-guided models to turn <i>our</i> cat into a painting, or imagine a new product based on <i>our</i> favorite toy?
  Here we present a simple approach that allows such creative freedom.
  Using only 3-5 images of a user-provided concept, like an object or a style, we learn to represent it through new "words" in the embedding space of a frozen text-to-image model.
  These "words" can be composed into natural language sentences, guiding <i>personalized</i> creation in an intuitive way.
  Notably, we find evidence that a <i>single</i> word embedding is sufficient for capturing unique and varied concepts.
  We compare our approach to a wide range of baselines, and demonstrate that it can more faithfully portray the concepts across a range of applications and tasks.

## Citation for Textual Inversion

The Textual Inversion repo asks to cite the paper. Information can be seen below:

```
@misc{gal2022textual,
      doi = {10.48550/ARXIV.2208.01618},
      url = {https://arxiv.org/abs/2208.01618},
      author = {Gal, Rinon and Alaluf, Yuval and Atzmon, Yuval and Patashnik, Or and Bermano, Amit H. and Chechik, Gal and Cohen-Or, Daniel},
      title = {An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion},
      publisher = {arXiv},
      year = {2022},
      primaryClass={cs.CV}
}
```

## Results
Here are some sample results. Please visit our [project page](https://textual-inversion.github.io/) or read our paper for more!

![](img/teaser.jpg)

![](img/samples.jpg)

![](img/style.jpg)
