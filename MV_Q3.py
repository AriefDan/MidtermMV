import cv2
from ultralytics import YOLO

model = YOLO('Arief.pt')

vid = cv2.VideoCapture(0) #Can use video using title ie 'Video.mp4', 0 to call directly from webcam

while True:

   ret, frame = vid.read()

   results = model(frame, stream=True) #Feedforwarding the fram through the model, geeting the result

   for r in results:
      boxes = r.boxes
      for bbox in boxes:
         x1,y1,x2,y2 = bbox.xyxy[0]
         x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

         cls_idx = int(bbox.cls[0])
         cls_name = model.names[cls_idx]

         conf = round(float(bbox.conf[0]),2)


         cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),4)
         cv2.putText(frame,f'{cls_name} {conf}', (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

   cv2.imshow('My Pic',frame) 
   
   cv2.waitKey(1)

vid.release()
cv2.destroyAllWindows() #Use CTRL + C to stop videocapture