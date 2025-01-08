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



def detect_with_camera(model_path, types_of_worms, exact_worm_count):
  
  def delay_seconds_for_camera(time_delay):
    global worm_seen  
    print("delay start")
    sleep(time_delay)
    worm_seen = False
    print("time over")

  directory_name = datetime.now().strftime('%b-%d-%Y--%H-%M-%S')
  os.mkdir(directory_name)

  if not isinstance(types_of_worms, list):
    raise TypeError(f"\033[91m{types_of_worms} input must be an array (list).\033[0m"
                    f"\033[92m Example of a valid input: `[larva_worm_str, adult_worm_str]`\033[0m")


  # How to capture data from webcam on Windows
  cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

  num_frames_to_record = 1000

  fps_for_vid_output = 1 # Value to be increased

  # Make sure the camera is available
  assert cap.isOpened()
  
  serial1=serial.Serial ("COM1",9600)

  serial1.write(b'\xee')

  # cuda means we will use GPU, otherwise we will be using CPU
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  torch.set_printoptions(sci_mode=False)

  # Load our model
  model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
  model.to(device)

  window_width = 1000
  window_height = 1000
  output_video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps_for_vid_output, (window_width, window_height))

  screenshot_count = 1

  # CSV file will be created and named as "output.csv"
  with open("output.csv", mode='w') as csvfile:
    # Headers of the csv file
    fieldnames = ['Confidence', 'Time', 'Shutter On', 'Objects Detected']
    thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if csvfile.tell() == 0:
      thewriter.writeheader()
    

    while num_frames_to_record > 0:
      # Start time counter
      start_time = time()

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


      # Data to be be outputted to .csv file
      confidence_value = list(results.pandas().xyxy[0].to_dict()['confidence'].values())
      names = list(results.pandas().xyxy[0].to_dict()['name'].values()) 

      if worm_seen:
        shutter_on = True
      else:
        shutter_on = False
      
      # A list of what is detected in the frame
      show_labels = list((results.pandas().xyxy[0].to_dict()['name']).values()) 

      number_of_worms_detected = sum(label in show_labels for label in types_of_worms)

      # Put on the frame how many seconds has elapsed
      time_elapsed = round((time() - start_time), 3)
      time_elapsed_text = f"Time elapsed: {time_elapsed: .2f}s"
      cv2.putText(frame ,time_elapsed_text, (120,120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)


      # if worm shutter is on, put "Worm Seen" on the frame
      if worm_seen:
        cv2.putText(frame ,"Worm Seen", (20,200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
      
      # Draw a rectangle around every detection in the results
      for index, result in enumerate(results.xyxy[0]):
        x1 = result[0]
        y1 = result[1]
        x2 = result[2]
        y2 = result[3]
        confidence = result[4]


        worm_confidence = results.pandas().xyxy[0].to_dict()['confidence'][index]
        label_name = results.pandas().xyxy[0].to_dict()['name'][index]


        # Show text if one worm is detected but the confidence level is at least 0.3
        if number_of_worms_detected == exact_worm_count and label_name in types_of_worms and worm_confidence >= 0.3:
          
          result_text = f"# of objects seen (current frame): {(len(results.xyxy[0]))}"
          cv2.putText(frame ,result_text, (300,850), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
          start_point = (int(x1), int(y1))
          end_point = (int(x2), int(y2))
          cv2.rectangle(frame, start_point, end_point, red_colour, thickness)
        
          # Display confidence value in real-time
          confidence_to_text = f"{confidence: .2f}"
          cv2.putText(frame ,confidence_to_text, start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)


          if not worm_seen and worm_confidence >= 0.7:
            # Take a screenshot
            cv2.imwrite(f'screenshot{screenshot_count}.png', frame)
            screenshot_count += 1

            cv2.putText(frame ,str(show_labels), (300,900), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
            worm_seen = True
            serial1.write(b'\xaa')  
            print("shutter open")
            sleep(0.08)
            serial1.write(b'\xac')
            print("shutter close")
            time_delay = 10
            Thread(target=delay_seconds_for_camera, daemon=True, args=(time_delay,)).start()
            cv2.rectangle(frame, start_point, end_point, (255,0,0), thickness)
            
          if worm_seen:
            continue

      cv2.imshow('Worm Detection', frame)

      # Loop through the data of each detection when the algorithm sees any worm
      for i in range(len((results.pandas().xyxy[0].to_dict()['confidence']).values())):
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
  print("done!")
  print(f"time_elapsed: {total_time}")

  rows = []
  with open("output.csv", mode='r') as csvfile:
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
    plt.title('Data Points Aligned along X-axis')
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
  print(f"directory_name: \n\n{directory_name}")
  os.mkdir(directory_name)

  # How to capture data from video file
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

  torch.set_printoptions(sci_mode=False)

  # Load our model
  model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
  model.to(device)

  window_width = 1000
  window_height = 1000
  output_video_path = os.path.join(directory_name, 'output.mp4')
  output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_for_vid_output, (window_width, window_height))

  screenshot_count = 1

  # CSV file will be created and named as "output.csv"
  csv_outout_path = os.path.join(directory_name, 'output.csv')
  with open(csv_outout_path, mode='w') as csvfile:
    worm_seen = False
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

      # YOLOv5 uses RGB colour format for their images whereas OpenCV uses BGR, so we need to convert
      # between them to ensure we get the right results.
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      results = model(frame)
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

      # Red colour in BGR format
      red_colour = (0, 0, 255)
      thickness = 2

      # A list of what is detected in the frame
      show_labels = list((results.pandas().xyxy[0].to_dict()['name']).values()) 

      number_of_worms_detected = sum(label in show_labels for label in types_of_worms)

      # Put on the frame how many seconds has elapsed

      time_elapsed_text = f"Time elapsed: {total_time-(num_frames_to_record/fps_for_vid_output): .2f}s"
      cv2.putText(frame ,time_elapsed_text, (120,120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

      # Draw a rectangle around every detection in the results
      for index, result in enumerate(results.xyxy[0]):
        x1 = result[0]
        y1 = result[1]
        x2 = result[2]
        y2 = result[3]
        confidence = result[4]


        worm_confidence = results.pandas().xyxy[0].to_dict()['confidence'][index]
        label_name = results.pandas().xyxy[0].to_dict()['name'][index]


        # Show text if one worm is detected but the confidence level is at least 0.3
        if number_of_worms_detected == exact_worm_count and label_name in types_of_worms and worm_confidence >= 0.7:
          
          result_text = f"# of objects seen (current frame): {(len(results.xyxy[0]))}"
          cv2.putText(frame ,result_text, (300,850), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
          start_point = (int(x1), int(y1))
          end_point = (int(x2), int(y2))
          cv2.rectangle(frame, start_point, end_point, red_colour, thickness)
        
          # Display confidence value in real-time
          confidence_to_text = f"{confidence: .2f}"
          cv2.putText(frame ,confidence_to_text, start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

          shutter_on = True
          
          # Take a screenshot
          screenshot_path = os.path.join(directory_name, f'screenshot{screenshot_count}.png')
          cv2.imwrite(screenshot_path, frame)
          screenshot_count += 1
          
          cv2.putText(frame ,str(show_labels), (300,900), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
          print("shutter open")
          cv2.putText(frame ,"Worm Seen", (20,200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
          sleep(0.08)
          print("shutter close")
          cv2.rectangle(frame, start_point, end_point, (255,0,0), thickness)
          
      cv2.imshow('Worm Detection', frame)


      # Data to be be outputted to .csv file
      confidence_value = list(results.pandas().xyxy[0].to_dict()['confidence'].values())
      names = list(results.pandas().xyxy[0].to_dict()['name'].values()) 

        

      time_elapsed = round(total_time-(num_frames_to_record/fps_for_vid_output), 3)
      # Loop through the data of each detection when the algorithm sees any worm
      for i in range(len((results.pandas().xyxy[0].to_dict()['confidence']).values())):
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
  print(f"time_elapsed: {time_elapsed_text}")

  rows = []
  with open(csv_outout_path, mode='r') as csvfile:
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
    plt.title('Data Points Aligned along X-axis')
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
model_path = 'best_orig.pt' 
larva_worm_str = 'larva_worm'
adult_worm_str = 'adult_worm'



# Uncomment this line if using worm detection algorithm on camera
#with_camera(model_path, [larva_worm_str, adult_worm_str], 1)

# The function `detect_with_video` takes the third parameter only as an array
# Uncomment the next two lines if using an input video
video_name = "10s.mp4"
detect_with_video(video_name, model_path,[adult_worm_str], 1)
