import torch
import cv2
from time import sleep
from time import time
from threading import Thread
import serial
import csv
from datetime import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator



def detect_with_camera(model_path, types_of_worms, exact_worm_count):
  global shutter_on  
  shutter_on = False
  
  def delay_seconds_for_camera(time_delay):
    global shutter_on  
    print("delay start")
    sleep(time_delay)
    shutter_on = False
    print("time over")

  directory_name = datetime.now().strftime('%b-%d-%Y--%H-%M-%S')
  os.mkdir(directory_name)

  if not isinstance(types_of_worms, list):
    raise TypeError(f"\033[91m{types_of_worms} input must be an array (list).\033[0m"
                    f"\033[92m Example of a valid input: `[larva_worm_str, adult_worm_str]`\033[0m")


  # How to capture data from webcam on Windows
  cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

  num_frames_to_record = 1000

  fps_for_vid_output = 11.41 # Match with camera fps

  # Make sure the camera is available
  assert cap.isOpened()
  
  serial1=serial.Serial ("COM1",9600)

  serial1.write(b'\xee')

  # cuda means we will use GPU, otherwise we will be using CPU
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  torch.set_printoptions(profile="short", sci_mode=False)

  # Load our YOLOv8 model
  model = YOLO(model_path)
  model.to(device)  

  # Defining popup window size
  window_width = 1000
  window_height = 1000

  output_video_path = os.path.join(directory_name, 'output.mp4')
  output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_for_vid_output, (window_width, window_height))

  screenshot_count = 1

  # CSV file will be created and named as "output.csv"
  csv_output_path = os.path.join(directory_name, 'output.csv')
  with open(csv_output_path, mode='w') as csvfile:
    # Headers of the csv file
    fieldnames = ['Confidence', 'Time', 'Shutter On', 'Objects Detected']
    thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if csvfile.tell() == 0:
      thewriter.writeheader()
    
    shutter_on = False

    # Start time counter
    start_time = time()

    while num_frames_to_record > 0:

      # Get the current frame
      ret, frame = cap.read()
      assert ret
      
      frame = cv2.resize(frame, (window_width, window_height))

      # YOLOv8 uses RGB colour format for their images whereas OpenCV uses BGR, so we need to convert
      # between them to ensure we get the right results.
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      
      # img needed to show the bounded boxes in the frame
      img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

      results = model.predict(img)
      
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)   
       

      if shutter_on:
        # if worm shutter is on, keep "Worm Seen" on the frame
        cv2.putText(frame ,"Worm Seen", (25,200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 4)
      else:
        shutter_on = False
      
      # A list of what is detected in the frame
      show_labels = [results[0].names[int(c.cls)] for c in results[0].boxes]
      number_of_worms_detected = sum(label in show_labels for label in types_of_worms)


      # Put on the frame how many seconds has elapsed
      time_elapsed_text = f"Time elapsed: {(time()-start_time): .2f}s"
      text_parts = time_elapsed_text.split(":")
      text_elapsed = text_parts[0] + ":" + text_parts[1]  # "Time Elapsed" part
      text_x, text_y = 25, 50
      
      # Add a dark background to the frame to improve visibility of the "Time Elapsed" text
      cv2.rectangle(frame, (20, text_y - 30), (text_x +10+ len(text_elapsed) * 15, text_y + 10), (5, 5, 5, 10), -1)
      
      # Add the "Time Elapsed" text to the frame
      cv2.putText(frame ,time_elapsed_text, (25,50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)


      # Loop through each object that is detected
      for box in (results[0].boxes):

        result = box.data.tolist()[0]
        x1 = result[0]
        y1 = result[1]
        x2 = result[2]
        y2 = result[3]
        
        confidence = result[4]

        annotator = Annotator(frame)
        label_name = model.names[int(result[5])]
        annotator_text = f"{model.names[int(result[5])]} {round(confidence,2)}"


        # Show text if one worm is detected but the confidence level is at least 0.7
        if number_of_worms_detected == exact_worm_count and label_name in types_of_worms and confidence >= 0.7 and not shutter_on:  
          # Set `shutter_on` to True so output.csv writes that the shutter was on at this frame
          shutter_on = True

          # Add a bounding box to the detected object
          BGR_orange = (0,165,255)
          BGR_blue = (255,144,30)

          if label_name == "larva_worm":
            annotator.box_label([x1,y1,x2,y2] ,annotator_text, BGR_orange)
          elif label_name == "adult_worm":
            annotator.box_label([x1,y1,x2,y2] ,annotator_text, BGR_blue)
          
          # Take a screenshot
          screenshot_path = os.path.join(directory_name, f'screenshot{screenshot_count}.png')
          cv2.imwrite(screenshot_path, frame)
          screenshot_count += 1
          
          # Put "Worm Seen" Text in frame
          cv2.putText(frame ,"Worm Seen", (25,200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 4)
          serial1.write(b'\xaa')  # Open Shutter
          print("shutter open")
          sleep(0.08)
          serial1.write(b'\xac') # Close Shutter
          print("shutter close")
          time_delay = 10
          # Create another thread to prevent another worm detection for `time_delay` seconds
          Thread(target=delay_seconds_for_camera, daemon=True, args=(time_delay,)).start()


      cv2.imshow('Worm Detection', frame)

      # Data to be be outputted to .csv file
      confidence_value = [confidence for c in results[0].boxes]
      names = [model.names[results[0].boxes.data.tolist()[0][5]] for c in results[0].boxes]
      time_elapsed = round((time()-start_time), 3)

      # Loop through the data of each detection when the algorithm sees any worm
      for i in range(len(show_labels)):
        # Write the confidence score, time, if shutter is on/off and what objects are detected to the "output.csv" file.
        thewriter.writerow({'Confidence' : round(confidence_value[i], 3), 'Time' : time_elapsed, 'Shutter On' : shutter_on, 'Objects Detected' : names[i]})
      

      # waitKey() is needed otherwise imshow() does not work correctly
      cv2.waitKey(int(fps_for_vid_output))

      # Write the current frame
      output_video.write(frame)

      num_frames_to_record = num_frames_to_record - 1

  cap.release()
  output_video.release() # save video

  cv2.destroyAllWindows()
  total_time = time_elapsed
  print(f"time_elapsed: {total_time}")

  rows = []
  with open(csv_output_path, mode='r') as csvfile:
    reader = csv.DictReader(csvfile)
    rows = list(reader)

  # Filter rows with "Shutter On" as "TRUE" after a "FALSE" row
  filtered_rows = []
  previous_row_shutter_on = False
  for row in rows:
      current_row_shutter_on = row["Shutter On"] == "True"
      if previous_row_shutter_on == False and current_row_shutter_on:
          filtered_rows.append(row)
      previous_row_shutter_on = current_row_shutter_on

  # Write the filtered rows to a new CSV file
  if len(filtered_rows) > 0:
    file_path = os.path.join(directory_name, "worm_detections.csv")
    with open(file_path, mode='w', newline='') as csvfile:
      fieldnames = list(filtered_rows[0].keys())
      writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
      writer.writeheader()
      writer.writerows(filtered_rows)
    print("Filtered rows written to worm_detections.csv")
    
    ### Begin plotting worm detections
    file_path = os.path.join(directory_name, 'worm_detections.csv')
    df = pd.read_csv(file_path, delimiter=',')

    # Convert the 'Time' column to numeric values
    df['Time'] = pd.to_numeric(df['Time'])

    # Create a single y-value for all data points
    y_value = 0

    # Plot the data points as a scatter plot
    plt.scatter(df['Time'], [y_value] * len(df), color='blue', marker='o')
    plt.xlabel('Time (s)')
    plt.title('Worm Detections')
    plt.yticks([])  # Remove the y-axis ticks and labels
    plt.box(False)
    plt.grid(False)

    plt.xlim(0, total_time)

    # Save the plot as an image file (e.g., PNG)
    plt.savefig(os.path.join(directory_name, 'plot_detections.png'), dpi = 500)

  else:
    print("No matching rows found, worm_detections.csv is not created")
    print("plot_detections.png is not created because your specified detections were not found")
  print("done!")



def detect_with_video(video_name, model_path, types_of_worms, exact_worm_count):

  directory_name = f"{datetime.now().strftime('%b-%d-%Y--%H-%M-%S')} ({video_name})"
  os.mkdir(directory_name)

  # Capture data from video file
  if os.path.exists(video_name):
    cap = cv2.VideoCapture(video_name) # Capture from file
  else:
    raise FileNotFoundError("\033[91mThe file path you have entered is not valid.\033[0m")


  if not isinstance(types_of_worms, list):
    raise TypeError(f"\033[91m{types_of_worms} input must be an array (list).\033[0m"
                  f"\033[92m Example of a valid input: `[larva_worm_str, adult_worm_str]`\033[0m")

  num_frames_to_record = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # record all frames from video (if using file)

  fps_for_vid_output = (cap.get(cv2.CAP_PROP_FPS)) # match output fps to input video fps (if using file)

  # cuda means we will use GPU, otherwise we will be using CPU
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  torch.set_printoptions(profile="short", sci_mode=False)

  # Load our YOLOv8 model
  model = YOLO(model_path)
  model.to(device)

  # Defining popup window size
  window_width = 1000
  window_height = 1000
  
  output_video_path = os.path.join(directory_name, 'output.mp4')
  output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_for_vid_output, (window_width, window_height))

  screenshot_count = 1
  
  # CSV file will be created and named as "output.csv"
  csv_output_path = os.path.join(directory_name, 'output.csv')
  with open(csv_output_path, mode='w') as csvfile:
    # Headers of the csv file
    fieldnames = ['Confidence', 'Time', 'Shutter On', 'Objects Detected']
    thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if csvfile.tell() == 0:
      thewriter.writeheader()
    total_time = num_frames_to_record/fps_for_vid_output

    while num_frames_to_record > 0:
      shutter_on = False

      # Get the current frame
      ret, frame = cap.read()
      assert ret

      frame = cv2.resize(frame, (window_width, window_height))
      
      # YOLOv8 uses RGB colour format for their images whereas OpenCV uses BGR, so we need to convert
      # between them to ensure we get the right results.
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

      results = model.predict(img)
      
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


      # A list of what is detected in the frame
      show_labels = [results[0].names[int(c.cls)] for c in results[0].boxes]
      number_of_worms_detected = sum(label in show_labels for label in types_of_worms)

      # Put on the frame how many seconds has elapsed
      time_elapsed_text = f"Time elapsed: {total_time-(num_frames_to_record/fps_for_vid_output): .2f}s"
      text_parts = time_elapsed_text.split(":")
      text_elapsed = text_parts[0] + ":" + text_parts[1]  # "Time Elapsed" part
      text_x, text_y = 25, 50

      # Add a dark background to the frame to improve visibility of the "Time Elapsed" text
      cv2.rectangle(frame, (20, text_y - 30), (text_x +10+ len(text_elapsed) * 15, text_y + 10), (5, 5, 5, 10), -1)
      
      # Add the "Time Elapsed" text to the frame
      cv2.putText(frame ,time_elapsed_text, (25,50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

      # Loop through each object that is detected
      for box in (results[0].boxes):
        
        result = box.data.tolist()[0]
        
        x1 = result[0]
        y1 = result[1]
        x2 = result[2]
        y2 = result[3]
        
        confidence = result[4]
        
        annotator = Annotator(frame)
        label_name = model.names[int(result[5])]
        annotator_text = f"{model.names[int(result[5])]} {round(confidence,2)}"
        

        # Show text if one worm is detected but the confidence level is at least 0.7
        if number_of_worms_detected == exact_worm_count and label_name in types_of_worms and confidence >= 0.7: 
          # Set `shutter_on` to True so output.csv writes that the shutter was on at this frame
          shutter_on = True

          # Add a bounding box to the detected object
          BGR_blue = (255,144,30)
          BGR_orange = (0,165,255)

          if label_name == "larva_worm":
            annotator.box_label([x1,y1,x2,y2] ,annotator_text, BGR_orange)
          elif label_name == "adult_worm":
            annotator.box_label([x1,y1,x2,y2] ,annotator_text, BGR_blue)

          # Put "Worm Seen" Text in frame
          cv2.putText(frame ,"Worm Seen", (25,200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 4)
          
          # Take a screenshot
          screenshot_path = os.path.join(directory_name, f'screenshot{screenshot_count}.png')
          cv2.imwrite(screenshot_path, frame)
          screenshot_count += 1          
          print("shutter open")
          sleep(0.08)
          print("shutter close")
     
      cv2.imshow('Worm Detection', frame)
      
      
      # Data to be be outputted to .csv file
      confidence_value = [confidence for c in results[0].boxes]
      names = [model.names[results[0].boxes.data.tolist()[0][5]] for c in results[0].boxes]
      time_elapsed = round(total_time-(num_frames_to_record/fps_for_vid_output), 3)

      # Loop through the data of each detection when the algorithm sees any worm
      for i in range(len(show_labels)):
        # Write the confidence score, time, if shutter is on/off and what objects are detected to the "output.csv" file.
        thewriter.writerow({'Confidence' : round(confidence_value[i], 3), 'Time' : time_elapsed, 'Shutter On' : shutter_on, 'Objects Detected' : names[i]})
      

      # waitKey() is needed otherwise imshow() does not work correctly
      cv2.waitKey(int(fps_for_vid_output))

      # Write the current frame
      output_video.write(frame)

      num_frames_to_record = num_frames_to_record - 1

  cap.release()
  output_video.release() # save video

  cv2.destroyAllWindows()
  
  rows = []
  with open(csv_output_path, mode='r') as csvfile:
    reader = csv.DictReader(csvfile)
    rows = list(reader)
  
  # Filter rows with "Shutter On" as "TRUE"
  filtered_rows = []
  for row in rows:
      if row["Shutter On"] == "True":
        filtered_rows.append(row)

  ### Write the filtered rows to a new CSV file
  if len(filtered_rows) > 0:
    file_path = os.path.join(directory_name, 'worm_detections.csv')
    with open(file_path, mode='w', newline='') as csvfile:
      fieldnames = list(filtered_rows[0].keys())
      writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
      writer.writeheader()
      writer.writerows(filtered_rows)
    print("Filtered rows written to worm_detections.csv")

    ### Begin plotting worm detections
    file_path = os.path.join(directory_name, 'worm_detections.csv')
    df = pd.read_csv(file_path, delimiter=',')

    # Convert the 'Time' column to numeric values
    df['Time'] = pd.to_numeric(df['Time'])

    # Create a single y-value for all data points
    y_value = 0

    # Plot the data points as a scatter plot
    plt.scatter(df['Time'], [y_value] * len(df), color='blue', marker='o')
    plt.xlabel('Time (s)')
    plt.title('Worm Detections')
    plt.yticks([])  # Remove the y-axis ticks and labels
    plt.box(False)
    plt.grid(False)

    plt.xlim(0, total_time)

    # Save the plot as an image file (e.g., PNG)
    plt.savefig(os.path.join(directory_name, 'plot_detections.png'), dpi = 500)

  else:
    print("No matching rows found, worm_detections.csv is not created")
    print("plot_detections.png is not created because your specified detections were not found")
  print("done!")



# ********************************************
# ********************************************
# *****      ONLY Modify code below      *****
# ********************************************
# ********************************************



# The path to the ML model that the worm detection algorithm will use
model_path = 'best_yolov8_fold1.pt' 

larva_worm_str = 'larva_worm'
adult_worm_str = 'adult_worm'

video_name = "HighWormLoading.mp4"

# The function `detect_with_camera` takes the second parameter only as an array
### Uncomment this line if using worm detection algorithm on camera
#detect_with_camera(model_path, [larva_worm_str, adult_worm_str], 1)


# The function `detect_with_video` takes the third parameter only as an array
### Uncomment the next two lines if using an input video
detect_with_video(video_name, model_path,[larva_worm_str, adult_worm_str], 1)
