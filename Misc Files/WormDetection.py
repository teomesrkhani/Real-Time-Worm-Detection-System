import torch
import cv2
from time import sleep
from time import time
from threading import Timer
import serial

#### import timeit

is_delaying = False

def delay_seconds(time):
  global is_delaying

  if len(results.xyxy[0]) == 8:
    sleep(2)
    print("time over")

def perform_action():
    global is_delaying
    #print(f"{confidence}, {len(results.xyxy[0])} {results.xyxy[0]} yes")
    #for result in results.xyxy[0]:
      #confidence = result[4]
      #print(confidence)
    
    
    #for i in results.xyxy[0]:
      #print(i[4])
    print(len(results.xyxy[0]))
    t = time()
    print(t)
    print("******finished")
    if len(results.xyxy[0]) == 11:
      print(f"open: {time()}")
      sleep(0.005)
      print(f"close: {time()}")
      print("open shutter*****************")
    print("-----------")

    is_delaying = False

# How to capture data from webcam on Windows
#cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

# How to capture data from video file
cap = cv2.VideoCapture("Adult worm 2.mp4") # Capture from file

#num_frames_to_record = 100
num_frames_to_record = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # record all frames from video (if using file)

fps_for_vid_output = 1
#fps_for_vid_output = int(cap.get(cv2.CAP_PROP_FPS)) # match output fps to input video fps (if using file)

# Make sure the camera is available
assert cap.isOpened()

# cuda means we will use GPU, otherwise we will be using CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.set_printoptions(sci_mode=False)

# Load our model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='worm_model_0523.pt', force_reload=True)
model.to(device)

window_width = 800
window_height = 800

output_video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps_for_vid_output, (window_width, window_height))

#serial1 = serial.Serial("COM1",9600)

while num_frames_to_record > 0:
    # Get the current frame
    ret, frame = cap.read()
    assert ret
    
    frame = cv2.resize(frame, (window_width, window_height))

    # YOLOv5 uses RGB colour format for their images whereas OpenCV uses BGR, so we need to convert
    # between them to ensure we get the right results.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Red colour in BGR format
    red_colour = (0, 0, 255)
    thickness = 2

    
    #print(f"\n{filter(results.pandas().xyxy[0].to_dict()['name'], 'larva_worm')}\n")
    a = (results.pandas().xyxy[0].to_dict()['name'])

    #print((a))

    b = list((results.pandas().xyxy[0].to_dict()['name']).values())
    print(f"name:{b}\ncount: {b.count('larva_worm')}")
    
    # Draw a rectangle around every detection in the results
    for result in results.xyxy[0]:
      x1 = result[0]
      y1 = result[1]
      x2 = result[2]
      y2 = result[3]
      confidence = result[4]
      result_text = f"# of worms seen (current frame): {(len(results.xyxy[0]))}"
      cv2.putText(frame ,result_text, (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

      start_point = (int(x1), int(y1))
      end_point = (int(x2), int(y2))
      cv2.rectangle(frame, start_point, end_point, red_colour, thickness)
    
      # Display confidence value in real-time
      confidence_to_text = f"{confidence: .2f}"
      cv2.putText(frame ,confidence_to_text, start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    
      #print(result)

      """
      if confidence >= 0.9 and len(results.xyxy[0]) == 8:
      #if confidence >= 0.9 and len(results.xyxy[0]) == 5:
        #sleep(2)
        #if len(results.xyxy[0]) == 1:
          #serial1.write(b'\xaa')
          #serial.write(b'\xac')


        start = (time()*1000)
        sleep(0.05)
        end=(time()*1000)
        print(f"time 1: {end - start}")
        start = time()*1000
        end=time()*1000
        print(f"time 2: {end - start}")

        if not is_delaying:
          is_delaying = True
          #Timer(delay_seconds(2)
          Timer(4, perform_action).start()
          print("started")
          #for i in results.xyxy[0]:
            #print(i[4])
          print(len(results.xyxy[0]))
          t = time()
          print(t)

          # wait for time completion
          #t.join()
          cv2.rectangle(frame, start_point, end_point, (255,0,0), thickness)

            
          
        if is_delaying:
          continue

        """


    cv2.imshow('Worm Detection', frame)

    # waitKey() is needed otherwise imshow() does not work correctly
    cv2.waitKey(5)

    # Write the current frame
    output_video.write(frame)

    num_frames_to_record = num_frames_to_record - 1

cap.release()
output_video.release() # save video

cv2.destroyAllWindows()
print("done!")
