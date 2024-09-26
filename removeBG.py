# Importing Required Modules 
from rembg import remove 
from PIL import Image, ImageOps
import torch
from torchvision import datasets, transforms

def crop_center(pil_img, crop_width, crop_height):
        img_width, img_height = pil_img.size
        return pil_img.crop(((img_width - crop_width) // 2,
                             (img_height - crop_height) // 2,
                             (img_width + crop_width) // 2,
                             (img_height + crop_height) // 2))


for gesture in ['P', 'R', 'S']:
    for i in range(500):
      input_path =  'newPRS/train/' + gesture + '/' + i + '.png'
      output_path = 'PRS_RemBG/train/' + gesture + '/' + i + '.png'
        
      # Processing the image 
      input = Image.open(input_path) 
      input = remove(input)
      input = ImageOps.grayscale(input)
      input = crop_center(input, 200, 200)
    
      transform = transforms.Compose([transforms.PILToTensor()])
    
      tensor = transform(input) 
    
      # Removing the background from the given Image 
      output = remove(input)
    
      #Saving the image in the given path
      output.save(output_path)


import shutil

# Define the source and destination paths
labels = {'P', 'R', 'S'}
samples = list(np.arange(0, 500, 10))
for label in labels:
  destination_path = "PRS_RemBG/" + label + '/'
  for sample in samples:
    sample = str(sample)
    source_path = "PRS_RemBG/"+ label + '/' + sample + '.png'
    # Move the file from the source folder to the destination folder
    shutil.move(source_path, destination_path)
