# Environment setting 
#### Python version
Python 3.8
#### Hardware:
NVidia GeForce GTX 1080 Ti, cuda 10.0
#### Install packages
```pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html```

#### Evalution
1. Download best.pt and put it in ```./training/put_model```. And put test data into the project directory.
 ```
 /final
 ├── training/ 
 │ ├── put_model/ 
 │ │ ├── best.pt    # Place model here
 │ │ ├── config.yaml 
 │ │ └── ... 
 ├── test/  # Place test data here 
 ├── 110550080_inference.py 
 ├── requirements.txt 
 ├── ...
 ```
2. Run ```python 110550080_inference.py```
   Output will be ```110550080_submission.csv```
