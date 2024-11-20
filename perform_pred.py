#%%
from ultralytics import YOLO
# %% Load a model
model = YOLO("yolov8n.pt")  # load our custom trained model

# %%
result = model(r"C:\Users\sahus\Downloads\dete.v1i.yolov8\test\images\Image1265_jpeg.rf.0348b18132d4114af8c4a264083cc552.jpg")
# %%
result
# %% command line run
# Standard Yolo
!yolo detect predict model=yolov8n.pt source="C:\Users\sahus\Downloads\dete.v1i.yolov8\test\images\Image1265_jpeg.rf.0348b18132d4114af8c4a264083cc552.jpg" conf=0.3 
# %% Masks 
!yolo detect predict model=C:\Users\sahus\Downloads\dete.v1i.yolov8\yolov8n.pt source="C:\Users\sahus\Downloads\dete.v1i.yolov8\test\images\Image1269_jpeg.rf.eb087e209e1fd33cfc31f0b4c9792c60.jpg" conf=0.3 

# %%
