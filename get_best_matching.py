import cv2
import numpy as np
from skimage.feature import hog
import os
import hashlib


hashed_values = [hashlib.sha256(str(i).encode()).hexdigest()[:20] for i in range(10)]


image10X_img_size  = (500,500)
root_folder = os.path.dirname(os.path.abspath(__file__))
refined_img_dir = os.path.join(root_folder,"ten_cow_process")
# refined_img_dir = r"C:\Users\shekh\Downloads\BTP\ten_cow_process"

class Image10x:
  def __init__(self,image_dir):
    self.image_dir = image_dir
    self.original_img = cv2.imread(self.image_dir)
    self.desired_size = image10X_img_size
    self.resized_img = self.resized_image()
    self.resized_grayscale_img = cv2.cvtColor(self.resized_img, cv2.COLOR_BGR2GRAY)
    self.model_input_images = []


  def resized_image(self):
    return cv2.resize(self.original_img,self.desired_size)

  def get_canny_edges_of_any_image(self,input_img):
    blurred_image = cv2.GaussianBlur(input_img, (5,5), 1)
    _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    canny_edges = cv2.Canny(thresholded_image, 120, 220)
    return canny_edges

  def dilate_any_image(self,input_img,no_of_iterations = 2,kernel_size = (2, 2)):
    dilated_img = cv2.dilate(input_img, np.ones(kernel_size, np.uint8), iterations=no_of_iterations)
    return dilated_img

  def erode_any_image(self,input_img,no_of_iterations = 2,kernel_size = (2, 2)):
    print("erode:",input_img.shape)
    eroded_img = cv2.erode(input_img, np.ones(kernel_size, np.uint8) , iterations=no_of_iterations)
    return eroded_img

  def get_clahe_image(self,clahe_input_img):
    clahe =  cv2.createCLAHE(clipLimit=500.0, tileGridSize=(16, 16))
    clahe_img = clahe.apply(clahe_input_img)
    return clahe_img

  def get_largest_component_only(self,img):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    largest_component_mask = (labels == largest_label).astype(np.uint8)
    result_image = cv2.bitwise_and(img, img, mask=largest_component_mask)
    return result_image

  def generate_images(self):
    simple_canny = self.get_canny_edges_of_any_image(self.resized_grayscale_img)
    self.model_input_images.append(simple_canny)
    simple_dilated_from_canny = self.dilate_any_image(simple_canny,3,(2,2))
    self.model_input_images.append(simple_dilated_from_canny)
    clahe_orig_gray = self.get_clahe_image(self.resized_grayscale_img)
    canny_from_clahe = self.get_canny_edges_of_any_image(clahe_orig_gray)
    mean_of_clahe = np.mean(clahe_orig_gray)
    new = clahe_orig_gray<mean_of_clahe
    final1 = (canny_from_clahe<10)*250-new*200
    final2 = new*150+canny_from_clahe
    self.model_input_images.append(final1)
    hog_features, hog_image1 = hog(final1, pixels_per_cell=(8, 8), cells_per_block=(4, 4), visualize=True)
    self.model_input_images.append(hog_image1)
    image1,image2,image3,image4 = self.model_input_images
    top_row = np.hstack((image1, image2))
    bottom_row = np.hstack((image3, image4))
    result_image = np.vstack((top_row, bottom_row))
    result_image = cv2.resize(result_image,(700,700))
    return final1



def get_best_matching(input_image):
    # sample = cv2.imread(input_image)
    sample1 = Image10x(image_dir=input_image).generate_images()
    sample = np.zeros((sample1.shape[0], sample1.shape[1], 3), dtype=np.uint8)
    sample[:, :, 0] = sample1
    sample[:, :, 1] = sample1
    sample[:, :, 2] = sample1  
    best_score = 0
    filename = None
    image = None
    hashed_val = None
    kp1,kp2,mp = None,None,None
    all_refined_processed_images = os.listdir(refined_img_dir)
    for i in range(10):
        file = all_refined_processed_images[i]
        print(file)
        # each_img = cv2.imread(os.path.join(refined_img_dir,file))
        rgb_image=  cv2.imread(os.path.join(refined_img_dir,file))
        sift = cv2.SIFT.create()

        keypoints_1,descriptors_1 = sift.detectAndCompute(sample,None)
        keypoints_2,descriptors_2 = sift.detectAndCompute(rgb_image,None)

        matches = cv2.FlannBasedMatcher({'algorithm':1,'trees':10},{}).knnMatch(descriptors_1,descriptors_2,k=2)
        match_points = []
        for p,q in matches:
           if p.distance < 0.6*q.distance:
              match_points.append(p)
        print("Match length:",len(matches),len(match_points))
        keypoints = min(len(keypoints_1),len(keypoints_2))
        if (len(match_points)/keypoints)*100 > best_score:
           best_score = (len(match_points)/keypoints)*100
           filename = os.path.join(refined_img_dir,file)
           image = rgb_image
           hashed_val = hashed_values[i]
           kp1,kp2,mp = keypoints_1,keypoints_2,match_points

    print("BEST MATCH:",filename)
    print("Score:",str(best_score))
    result = cv2.drawMatches(sample,kp1,image,kp2,mp,None)
    result = cv2.resize(result,(800,400))
    matches_result_file_name = "matched_result.jpg"
    cv2.imwrite(matches_result_file_name,result)
    return [filename.replace("ten_cow_process","ten_cow"),matches_result_file_name,hashed_val]

# get_best_matching(r"C:\Users\shekh\Downloads\cattle_0300_DJI_0124_edited.jpg")
