```python
import cv2
import os
```

```python
input_movie = cv2.VideoCapture("nh_mm2.avi")

fps = int(input_movie.get(cv2.CAP_PROP_FPS))
size = (int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH)), 
        int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fNUMS = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

print(fps, size, fNUMS)
```

```python
def timeformat(m):
    if len(str(m)) == 1:
        return("0{}".format(m))
    else:
        return("{}".format(m))
```

```python
pre_file = "frame"
frame_style = ".jpeg"
m_per_fps = fps * 60
f_index = 0

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# output_movie = cv2.VideoWriter("output.mp4", cv2.CAP_FFMPEG, fourcc, fps, size)
output_movie = cv2.VideoWriter("output.mp4", fourcc, fps, size, True)


while True:
    # Grab a single frame of video
    success, frame = input_movie.read()
 
    # Quit when the input video file ends
    if not success:
        break
    
    # frame_resize = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
    # Write the resulting image to the output video file
    # print("Writing frame {} / {}".format(frame_number, length))

    f_fps = f_index % fps
    f_sec = f_index // fps
    f_min = f_index // m_per_fps
    f_sec = f_sec % 60
    
    frame_label = [
        pre_file,
        timeformat(f_min),
        timeformat(f_sec),
        timeformat(f_fps)
    ]
    frame_name = "_".join(frame_label) + frame_style
    cv2.imwrite(os.path.join(figs_dir, frame_name), frame)


    new_frame = cv2.imread(os.path.join(figs_dir, frame_name))
    output_movie.write(new_frame)

    f_index += 1

    if f_index % m_per_fps == 0:
        print(
            "[INFO] The No.{} minute of the video has been processed!".format(
                timeformat(f_min)
            )
        )

print("[SUCCESS] The video has been processed!")
# All done!
input_movie.release()
cv2.destroyAllWindows()
```
