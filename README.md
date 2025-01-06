# IsEven ML
A PyTorch classification model to determine if a number is odd or even.

## How Do I Use This?
1. First, install the requirements with `pip install -r requirements.txt`.
2. Next, check if you have a GPU by running `test-cuda.py`. If you have a GPU, the training process will be much faster.
3. Next, add test data to `test_data.csv`. The model only works on natural numbers up to 65535 and 0.
4. Then, execute `main.py`. This will use `iseven-ml.pth`, delete this file if you'd like to train the model from scratch.
5. To test the model, run `test.py`.

## Tweaking Parameters
You can tweak the number of epochs that the model is trained on in `main.py`. Modify the `num_epochs` variable. Ensure that you delete `iseven-ml.pth` afterwards to re-generate the model.

## Gallery
![The command line output of the programme.](images/cli-output.png)
![The ROC curve from test.py.](images/roc-curve.png)
![CLI output from test.py.](images/test-cli-output.png)