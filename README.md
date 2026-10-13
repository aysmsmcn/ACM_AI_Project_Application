# ACM AI Project Application

I built a program that detects different types of clouds.

I chose to finetune a model using the dataset Clouds-1500 from Mendeley Data: <https://data.mendeley.com/datasets/2khchjbgzr/2>

I also chose to use the YOLO model from Ultralytics because I played around with it a little over the summer

All libraries used are listed in requirements.txt

What I did:
- First I converted the annotations in the dataset to YOLO formatted labels using the script convert.py
- I randomly split the photos in the dataset into a train-validate-test split of 70-15-15
- Then I trained the YOLO26-seg model in the finetuning.py script
  - I first ran a smoke test then the full training
  - One of the runs failed (it is labeled "cloud_n_640")
  - The real results are labeled "cloud_n_416"
- Results are saved in the runs folder
- I used the scripts evaluate.py and predict.py to quantitatively and qualitatively, respectively, analyze the model's results

Results:

Here are a list of observations I had:

1. A marginal amount of the results had stacked colors, meaning it classified the same clouds as two distinct types
2. There are many with misclassified results
3. Fragmentation does occur, but it usually matches the true outlines provided by the dataset's annotations
4. Many of the photos miss a few sparse clouds, especially photos taken during sunset/sunrise where the horizon is a different color
5. It is very good at distinctly classifying background elements (labled as Arvore)

Limitations:

Some limitations I had were time and computing power. I did this all on my laptop, which doesn't have a NVIDIA GPU, so all the training had to run through my CPU instead. This made training large amounts of epochs really slow and as a result, I had to reduce the number of epochs. I think this is a large reason why the model's accuracy wasn't quite there, as it was accurate for ~75% of the photos and that value can definitely be improved.

In the future, I'd improve it by just letting it train for a longer amount of time and get rid of some of tweaks I made to the training parameters to limit processing costs. For instance, I trained it on 416 pixels for pixel size and 30 epochs and in the future, I'd probably upscale the image size to 640 pixels and increase to 100 epochs for more rigorous training overall. I would also look into getting larger amounts of specifically sunset and sunrise data, as these photos were just really sparse overall in the dataset so I think the model wasn't able to pick up on them due to scarcity.

Sample Output:
<img width="670" height="504" alt="image" src="https://github.com/user-attachments/assets/19bfec7f-32a0-417f-8001-ef2c5854306e" />

The other outputs can be seen in the folder:
"Computer Vision Project\src\runs\segment\predict_test"

