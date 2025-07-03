# AI_Organoid_Counter
Uses a segmentation NN to count organoids in brightfield images.

Test the model on our [Huggingface Space!](https://huggingface.co/spaces/ReyaLabColumbia/ColonyCounter)

Fine-tuned on custom data from the following segmentation NN:
https://huggingface.co/nvidia/segformer-b3-finetuned-cityscapes-1024-1024

https://github.com/NVlabs/SegFormer?tab=readme-ov-file

Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). SegFormer: Simple and efficient design for semantic segmentation with transformers. arXiv preprint arXiv:2105.15203. https://arxiv.org/abs/2105.15203

The file ColonyAssaySegformerTern.py was used to download the model and fine-tune it using custom data. FIJI was used to create masks for the training data, and the file Image_Rotator.py was used to rotate, flip, and subdivide the large 1536x2048 images into 512x512 sections. The file Additional_training.py was used to fine-tune the model further. 

For the actual usage of the program, only the model itself and the following three Python files are needed:

Colony_analyzer_gui.py
Colony_analyzer_AI.py
Colony_analyzer_Zstack.py

<img width="572" alt="example" src="https://github.com/user-attachments/assets/739217d5-60bf-459b-a548-64d1ed42c316" />
<img width="572" alt="example2" src="https://github.com/user-attachments/assets/4d8a9ec4-f1f7-4bad-9ebe-16ba7b83b074" />

# Installation:
Have Python3, and the following libraries: Numpy, Pandas, Tkinter, Opencv, Transformers, PIL, matplotlib. NVIDIA's CUDA is recommended for performance but not strictly necessary for usage.
Download the three python files above and the model itself.

https://huggingface.co/ReyaLabColumbia/Segformer_Organoid_Counter_GP

Place the model subfolder and all the files in the same folder and it should be ready to run. To run in Windows, you can just left click Organoid_analyzer_gui.py. 

To set up a clickable icon in Linux, add the path to the Organoid_analyzer_gui.py file into the launch_gui.sh file (instructions inside) and then put the path to the launch_gui.sh file into the launcher.desktop file (instructions inside). Then put the launcher.desktop file on your /desktop and you can run it by clicking.

In any OS, you can run the program by left clicking inside the folder, open in terminal, and then python Organoid_analyzer_gui.py.

The following is an example of the output by the program:
![Group_analysis_results3](https://github.com/user-attachments/assets/cd5feeca-e5a1-40db-afbc-6f53d5f71f71)

[Group_analysis_results3.xlsx](https://github.com/user-attachments/files/20004750/Group_analysis_results3.xlsx)

