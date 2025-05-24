A work in progress aiming testing performance in native iOS environment with CoreML model inferences in echocardiography on segmentation in cardiac ultrasound, utilizing for testing purposes the integration of the work by "Van De Vyver, Gilles, et al. "Generative augmentations for improved cardiac ultrasound segmentation using diffusion models." arXiv preprint arXiv:2502.20100 (2025). https://arxiv.org/abs/2502.20100."

Original Repo README.md - From "https://github.com/GillesVanDeVyver/EchoGAINS" of the work by "Van De Vyver, Gilles, et al. "Generative augmentations for improved cardiac ultrasound segmentation using diffusion models." arXiv preprint arXiv:2502.20100 (2025). "

With also the contributions of:

"Dhariwal, Prafulla, and Alexander Nichol. "Diffusion models beat gans on image synthesis." Advances in neural information processing systems 34 (2021): 8780-8794."

"Lugmayr, Andreas, et al. "Repaint: Inpainting using denoising diffusion probabilistic models." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022."


# Echo Generative Augmentation using INpainting Synthesis (EchoGAINS) 🛜 💪

![Figure 1](figs/figure1.png)



## Publication & Citation

You should cite the following paper when using the code in this repository:

Van De Vyver, Gilles, et al. "Generative augmentations for improved cardiac ultrasound segmentation using diffusion models." arXiv preprint arXiv:2502.20100 (2025).
https://arxiv.org/abs/2502.20100

Blog post: [https://gillesvandevyver.com/#/projects/generative-ai](https://gillesvandevyver.com/#/projects/generative-ai)



## Toturial
A Jupyter notebook toturial is available at

[tutorial/tutorial.ipynb](tutorial/tutorial.ipynb).


## Installation
The package can be installed using pip from github:
```bash
pip install git+https://github.com/GillesVanDeVyver/EchoGAINS.git
```
Alternatively, you can download the source code and install it using pip:
```bash
pip install .
```

## Main features
EchoGAINS creates many variations of a labelled image without affecting annotation quality
An image and it's label are transformed and the missing parts are inpainted to create a
realistic sector again.
![Figure 2](figs/figure2.png)

A segmentation model trained on the augmented dataset is significantly more robust.

![Figure 3](figs/figure3.png)


## Demo
A demo comparing the segmentation performance of a nnU-Net model trained on the original dataset and the augmented dataset is available at:
[YouTube demo](https://www.youtube.com/watch?v=kiuWaPJnLHU&ab_channel=GillesVanDeVyver)

## License and Attribution
This project is an adaptation of the RePaint project, which itself is an adaptation from the Guided Diffusion project: 

Original project: guided-diffusion by Dhariwal and Nichol (OpenAI), available at https://github.com/openai/guided-diffusion, licensed under the MIT License.
Modifications were made in the RePaint project by Lugmayr et al. (Huawei Technologies Co., Ltd.), available at https://github.com/andreas128/RePaint, licensed under CC BY-NC-SA 4.0.
This project, EchoGAINS, by Van De Vyver et al. (Norwegian University of Science and Technology), is a modification of RePaint and is licensed under CC BY-NC-SA 4.0.

## Pretrained model
- Diffusion model trained on the CAMUS dataset: https://huggingface.co/gillesvdv/CAMUS_diffusion_model
- nnU-Net segmentation model trained on the augmented CAMUS dataset: https://huggingface.co/gillesvdv/nnU-Net_CAMUS_EchoGAINS


## Training your own model
- **Duffusion model**:

The generative diffusion model in this work was trained using the guided-diffusion package:

_Dhariwal, Prafulla, and Alexander Nichol. "Diffusion models beat gans on image synthesis." Advances in neural information processing systems 34 (2021): 8780-8794._

You can follow the instructions in the guided-diffusion repository to train your own model.
The pretrained model mentioned above was trained with the following parameters:
```
--image_size 256 --num_channels 64 --num_res_blocks 4 --learn_sigma True --diffusion_steps 4000 --noise_schedule cosine --lr 1e-4 --batch_size 32
```


- **Segmentation model**:

The segmentation model was trained using nnU-Net:

_Isensee, Fabian, et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." Nature methods 18.2 (2021): 203-211._

You can follow the instructions in the nnU-Net repository to train your own model.
The pretrained model mentioned above was trained with the default parameters of nnU-Net.


## Contact

Developer: <br />
[https://gillesvandevyver.com/](https://gillesvandevyver.com/)


Management: <br />
lasse.lovstakken@ntnu.no <br />
erik.smistad@ntnu.no <br />


The package has been developed using python 3.10.
