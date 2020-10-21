
This is the project of deeplearning course, we tried lsq and nips 2019 paper quantization technique in different ways.



+----+-------+--------------------------------------------------------------------+---------------------------+--------+
| #  | Model | Hyperparameters                                                    | Note                      | Acc@1  |
+----+-------+--------------------------------------------------------------------+---------------------------+--------+
| 1  | FP32  | lr=0.1, wd=1e-4, batch_size=64, epochs=90, lr_scheduler=cosine     | Training from scratch     | 90.650 |
+----+-------+--------------------------------------------------------------------+---------------------------+--------+
| 2  | FP32  | lr=0.1, wd=1e-4, batch_size=64, epochs=90, lr_scheduler=cosine     | Training from scratch     | 90.800 |
+----+-------+--------------------------------------------------------------------+---------------------------+--------+
| 3  | FP32  | lr=0.1, wd=1e-4, batch_size=64, epochs=90, lr_scheduler=cosine     | Training from scratch     | 92.170 |
+----+-------+--------------------------------------------------------------------+---------------------------+--------+
| 4  | W8A8  | lr=0.001, wd=1e-4, batch_size=64, epochs=1, lr_scheduler=cosine    | Initialized from model #1 | 90.571 |
+----+-------+--------------------------------------------------------------------+---------------------------+--------+
| 4  | W4a4  | lr=0.01, wd=1e-4, batch_size=64, epochs=90, lr_scheduler=cosine    | Initialized from model #1 | 90.850 |
+----+-------+--------------------------------------------------------------------+---------------------------+--------+
| 4  | W3A3  | lr=0.01, wd=0.5e-4, batch_size=64, epochs=90, lr_scheduler=cosine  | Initialized from model #1 | 90.490 |
+----+-------+--------------------------------------------------------------------+---------------------------+--------+
| 4  | W2A2  | lr=0.01, wd=0.25e-4, batch_size=64, epochs=90, lr_scheduler=cosine | Initialized from model #1 | 90.190 |
+----+-------+--------------------------------------------------------------------+---------------------------+--------+

To run this source, you need to install: pytorch & tensorboard.
For the Imagenette dataset, you can find the information and download link at: https://github.com/fastai/imagenette

Create a virtual environment

virtualenv -p python3 <env_name>
. <env_name>/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

