# AI_Colony_Counter
Uses a segmentation NN to count cell colonies

Fine-tuned on custom data from the following segmentation NN:
https://huggingface.co/nvidia/segformer-b3-finetuned-cityscapes-1024-1024

Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). SegFormer: Simple and efficient design for semantic segmentation with transformers. arXiv preprint arXiv:2105.15203. https://arxiv.org/abs/2105.15203

The file ColonyAssaySegformerTern.py was used to download the model and fine-tune it using custom data. FIJI was used to create masks for the training data, and the file Image_Rotator.py was used to rotate, flip, and subdivide the large 1536x2048 images into 512x512 sections. The file Additional_training.py was used to fine-tune the model further. 

For the actual usage of the program, only the model itself and the following three Python files are needed:

Mari_GUI_AI_3.py

Colony_Analyzer_AI_zstack2.py

Colony_Analyzer_AI2.py


<img width="572" alt="example" src="https://github.com/user-attachments/assets/739217d5-60bf-459b-a548-64d1ed42c316" />
<img width="572" alt="example2" src="https://github.com/user-attachments/assets/4d8a9ec4-f1f7-4bad-9ebe-16ba7b83b074" />

Installation:
Have Python3, and the following libraries: Numpy, Pandas, Tkinter, Opencv, Transformers, PIL, matplotlib. NVIDIA's CUDA is recommended for performance but not strictly necessary for usage.
Download the three python files above and the model itself.
Then change the paths in the python scripts so they call each other and the model.

You run the GUI file, and then select .tif files in the dropdown.

The following is an example of the output by the program:
![Group_analysis_results3](https://github.com/user-attachments/assets/cd5feeca-e5a1-40db-afbc-6f53d5f71f71)

[Group_analysis_results3.xlsx](https://github.com/user-attachments/files/20004750/Group_analysis_results3.xlsx)

