# # engine.py - FINAL VERSION WITH SHADOW REMOVAL

# import cv2
# import numpy as np
# import json
# from typing import Dict, List, Any

# def four_point_transform(image, pts):
#     rect = np.array(pts, dtype="float32")
#     (tl, tr, br, bl) = rect
#     widthA = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2)); widthB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1])**2))
#     maxWidth = max(int(widthA), int(widthB))
#     heightA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2)); heightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))
#     maxHeight = max(int(heightA), int(heightB))
#     dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]], dtype="float32")
#     M = cv2.getPerspectiveTransform(rect, dst)
#     warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
#     return warped

# def flatten_image(image_path: str):
#     image = cv2.imread(image_path)
#     if image is None: raise Exception(f"Could not load image: {image_path}")
    
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # --- NEW SHADOW REMOVAL LOGIC ---
#     # Apply a heavy blur to get the general illumination/shadow map
#     blurred_lighting = cv2.GaussianBlur(gray, (21, 21), 0)
#     # Divide the original grayscale image by the shadow map to normalize it
#     normalized = cv2.divide(gray, blurred_lighting, scale=255)
#     # --- END OF SHADOW REMOVAL ---

#     # All subsequent operations are performed on the SHADOW-FREE 'normalized' image
#     blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
#     _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
#     cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     if not cnts: raise Exception("No content found on the page.")
        
#     all_points = np.concatenate(cnts)
#     rect = cv2.minAreaRect(all_points)
#     box = cv2.boxPoints(rect)
#     box = np.intp(box)

#     warped = four_point_transform(image, box.reshape(4, 2))
    
#     (h, w) = warped.shape[:2]
#     if w > h: warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
#     return warped

# class OMRGrader:
#     def __init__(self, answer_key: Dict):
#         self.answer_key = answer_key
#         self.ANSWER_MAP = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
#         self.SUBJECTS = {"Python": (1, 20), "Data Analysis": (21, 40), "MySQL": (41, 60), "Power BI": (61, 80), "Adv Stats": (81, 100)}

#     # In engine.py, replace only the 'grade' method in the OMRGrader class

#     def grade(self, image: np.ndarray, set_name: str) -> Dict[str, Any]:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
#         blurred_lighting = cv2.GaussianBlur(gray, (21, 21), 0)
#         normalized = cv2.divide(gray, blurred_lighting, scale=255)
        
#         thresh = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
#         kernel = np.ones((3, 3), np.uint8)
#         opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1) # Only ONE iteration


#         # --- THIS IS THE DEBUG VISUALIZATION ---
#         print("Displaying the image after text removal. Press any key to continue...")
#         cv2.imshow("Cleaned Image (Text Removed)", cv2.resize(opening, (600, 800)))
#         cv2.waitKey(0)
#         # --- END OF DEBUG VISUALIZATION ---
        
#         cnts, _ = cv2.findContours(opening.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
#         bubble_contours = []
#         for c in cnts:
#             (x, y, w, h) = cv2.boundingRect(c)
#             if 0.6 <= w / float(h) <= 1.4 and 10 < w < 45 and 10 < h < 45:
#                 bubble_contours.append(c)

#         print(f"Found {len(bubble_contours)} bubbles on the cleaned, shadow-free image.")
        
#         # Draw what it found for a final check
#         final_debug_img = image.copy()
#         cv2.drawContours(final_debug_img, bubble_contours, -1, (0, 255, 0), 2)
#         cv2.imshow("Final Detected Bubbles", cv2.resize(final_debug_img, (600, 800)))
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#         if not (390 <= len(bubble_contours) <= 410):
#             raise Exception("Failed to detect correct number of bubbles.")

#         # ... The rest of the grading logic remains the same ...
#         bubble_contours.sort(key=lambda c: cv2.boundingRect(c)[0])
#         columns = [bubble_contours[i:i+80] for i in range(0, len(bubble_contours), 80)]
#         sorted_bubbles = []
#         for col in columns:
#             col.sort(key=lambda c: cv2.boundingRect(c)[1])
#             sorted_bubbles.extend(col)
        
#         questions = [sorted_bubbles[i:i+4] for i in range(0, len(sorted_bubbles), 4)]
#         key_to_use = self.answer_key.get(set_name, {})
#         subject_scores = {s: 0 for s in self.SUBJECTS}; total_correct, detailed_results = 0, []

#         for i, question_bubbles in enumerate(questions):
#             question_num_str = str(i + 1); is_correct, student_answer = False, "No Answer"
            
#             pixel_counts = [cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=cv2.drawContours(np.zeros(thresh.shape, dtype="uint8"), [b], -1, 255, -1))) for b in question_bubbles]
            
#             if max(pixel_counts) > 50:
#                 student_idx = np.argmax(pixel_counts); student_answer = chr(65 + student_idx)
#                 correct_char = key_to_use.get(question_num_str, "").lower(); correct_idx = self.ANSWER_MAP.get(correct_char)
#                 if student_idx == correct_idx:
#                     is_correct = True; total_correct += 1
#                     for sub, (start, end) in self.SUBJECTS.items():
#                         if start <= i + 1 <= end: subject_scores[sub] += 1; break
            
#             detailed_results.append({"question": question_num_str, "student_answer": student_answer, "is_correct": is_correct})
            
#         results = {"total_score": total_correct, "subject_scores": subject_scores, "detailed_results": detailed_results}
#         return results


# # engine.py - THE FINAL VERSION. NO MORE CHANGES.

# import cv2
# import numpy as np
# import json
# from typing import Dict, List, Any

# def flatten_image(image_path: str):
#     image = cv2.imread(image_path)
#     if image is None: raise Exception(f"Could not load image: {image_path}")
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
#     cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     if not cnts: raise Exception("No content found on the page.")
#     all_points = np.concatenate(cnts)
#     rect = cv2.minAreaRect(all_points)
#     box = cv2.boxPoints(rect); box = np.intp(box)
    
#     # Simple four_point_transform logic
#     pts = box.reshape(4, 2); rect = np.zeros((4, 2), dtype="float32")
#     s = pts.sum(axis=1); diff = np.diff(pts, axis=1)
#     rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
#     rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
#     (tl, tr, br, bl) = rect
#     widthA = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2)); widthB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1])**2))
#     maxWidth = max(int(widthA), int(widthB))
#     heightA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2)); heightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))
#     maxHeight = max(int(heightA), int(heightB))
#     dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]], dtype="float32")
#     M = cv2.getPerspectiveTransform(rect, dst)
#     warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

#     (h, w) = warped.shape[:2]
#     if w > h: warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
#     return warped

# class OMRGrader:
#     def __init__(self, answer_key: Dict):
#         self.answer_key = answer_key
#         self.ANSWER_MAP = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
#         self.SUBJECTS = {"Python": (1, 20), "Data Analysis": (21, 40), "MySQL": (41, 60), "Power BI": (61, 80), "Adv Stats": (81, 100)}

#     def grade(self, image: np.ndarray, set_name: str) -> Dict[str, Any]:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
#         cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
#         image_height = image.shape[0]
#         min_size = image_height * 0.015
#         max_size = image_height * 0.04

#         bubble_contours = []
#         for c in cnts:
#             (x, y, w, h) = cv2.boundingRect(c)
#             if 0.8 <= w / float(h) <= 1.2 and min_size < w < max_size and min_size < h < max_size:
#                 bubble_contours.append(c)

#         print(f"Found {len(bubble_contours)} bubbles.")
#         if not (390 <= len(bubble_contours) <= 410):
#             raise Exception("Failed to detect correct number of bubbles. Try a cleaner image.")

#         bubble_contours.sort(key=lambda c: cv2.boundingRect(c)[0])
#         columns = [bubble_contours[i:i+80] for i in range(0, len(bubble_contours), 80)]
#         sorted_bubbles = []
#         for col in columns:
#             col.sort(key=lambda c: cv2.boundingRect(c)[1])
#             sorted_bubbles.extend(col)
        
#         # Split into groups of 4, but the options are not yet sorted
#         questions_unsorted_options = [sorted_bubbles[i:i+4] for i in range(0, len(sorted_bubbles), 4)]
        
#         # --- THIS IS THE FINAL ONE-LINE FIX ---
#         # For each question, sort its 4 bubbles from left-to-right (by x-coordinate)
#         questions = [sorted(q, key=lambda c: cv2.boundingRect(c)[0]) for q in questions_unsorted_options]
        
#         key_to_use = self.answer_key.get(set_name, {})
#         subject_scores = {s: 0 for s in self.SUBJECTS}; total_correct, detailed_results = 0, []

#         for i, question_bubbles in enumerate(questions):
#             question_num_str = str(i + 1); is_correct, student_answer = False, "No Answer"
            
#             pixel_counts = [cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=cv2.drawContours(np.zeros(thresh.shape, dtype="uint8"), [b], -1, 255, -1))) for b in question_bubbles]
            
#             if max(pixel_counts) > (min_size * min_size * 0.3):
#                 student_idx = np.argmax(pixel_counts); student_answer = chr(65 + student_idx)
#                 correct_char = key_to_use.get(question_num_str, "").lower(); correct_idx = self.ANSWER_MAP.get(correct_char)
#                 if student_idx == correct_idx:
#                     is_correct = True; total_correct += 1
#                     for sub, (start, end) in self.SUBJECTS.items():
#                         if start <= i + 1 <= end: subject_scores[sub] += 1; break
            
#             detailed_results.append({"question": question_num_str, "student_answer": student_answer, "is_correct": is_correct})
            
#         results = {"total_score": total_correct, "subject_scores": subject_scores, "detailed_results": detailed_results}
#         return results



# engine.py - Enhanced OMR System with Comprehensive Bubble Detection

import cv2
import numpy as np
import json
from typing import Dict, List, Any, Tuple

def flatten_image(image_path: str, show_steps: bool = False):
    """
    Flatten and correct perspective of OMR sheet image
    """
    image = cv2.imread(image_path)
    if image is None: 
        raise Exception(f"Could not load image: {image_path}")
    
    if show_steps:
        cv2.imshow("1. Original Image", cv2.resize(image, (600, 800)))
        cv2.waitKey(0)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if show_steps:
        cv2.imshow("2. Grayscale", cv2.resize(gray, (600, 800)))
        cv2.waitKey(0)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding for better edge detection
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    if show_steps:
        cv2.imshow("3. Thresholded", cv2.resize(thresh, (600, 800)))
        cv2.waitKey(0)
    
    # Find contours to detect the document boundary
    cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: 
        raise Exception("No content found on the page.")
    
    # Filter contours by area to avoid small noise contours
    min_area = (image.shape[0] * image.shape[1]) * 0.1  # At least 10% of image area
    large_contours = [c for c in cnts if cv2.contourArea(c) > min_area]
    
    if not large_contours:
        # If no large contours found, use all contours
        large_contours = cnts
    
    # Get the largest contour (document boundary)
    all_points = np.concatenate(large_contours)
    rect = cv2.minAreaRect(all_points)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    # Order points: top-left, top-right, bottom-right, bottom-left
    pts = box.reshape(4, 2)
    rect_transform = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect_transform[0] = pts[np.argmin(s)]  # top-left
    rect_transform[2] = pts[np.argmax(s)]  # bottom-right
    rect_transform[1] = pts[np.argmin(diff)]  # top-right
    rect_transform[3] = pts[np.argmax(diff)]  # bottom-left
    
    # Calculate dimensions for perspective transform with padding to preserve content
    (tl, tr, br, bl) = rect_transform
    widthA = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2))
    widthB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1])**2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2))
    heightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Add padding to preserve more content and avoid cutting off edges
    padding_factor = 0.05  # 5% padding
    maxWidth = int(maxWidth * (1 + padding_factor))
    maxHeight = int(maxHeight * (1 + padding_factor))
    
    # Create destination points for perspective transform
    dst = np.array([[0, 0], [maxWidth - 1, 0], 
                   [maxWidth-1, maxHeight-1], [0, maxHeight-1]], dtype="float32")
    
    # Apply perspective transform
    M = cv2.getPerspectiveTransform(rect_transform, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # Improved rotation logic to preserve full image content
    # Only rotate if the aspect ratio suggests it's clearly landscape
    # and the rotation would improve the layout without cutting content
    h, w = warped.shape[:2]
    aspect_ratio = w / h
    
    # Only rotate if it's clearly landscape (ratio > 1.3) and would benefit from rotation
    if aspect_ratio > 1.3 and w > h:
        # Check if rotation would actually improve the layout
        # by ensuring we don't lose important content
        rotated = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        rotated_h, rotated_w = rotated.shape[:2]
        
        # Only use rotation if it results in a more reasonable aspect ratio
        # and doesn't significantly reduce the image area
        if rotated_w / rotated_h < 1.5:  # More portrait-like after rotation
            warped = rotated
    
    if show_steps:
        cv2.imshow("4. Flattened Image", cv2.resize(warped, (600, 800)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return warped

class OMRGrader:
    def __init__(self, answer_key: Dict):
        self.answer_key = answer_key
        self.ANSWER_MAP = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        self.SUBJECTS = {"Python": (1, 20), "Data Analysis": (21, 40), "MySQL": (41, 60), 
                        "Power BI": (61, 80), "Adv Stats": (81, 100)}

    def detect_bubble_area(self, image: np.ndarray, show_steps: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 1: Detect answer area using subject headings and crop the top part
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if show_steps:
            cv2.imshow("Step 1a: Grayscale Image", cv2.resize(gray, (600, 800)))
            cv2.waitKey(0)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use adaptive thresholding to better handle varying lighting
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        if show_steps:
            cv2.imshow("Step 1b: Adaptive Threshold", cv2.resize(thresh, (600, 800)))
            cv2.waitKey(0)
        
        # Detect subject headings to find answer area
        answer_start_y = self.detect_subject_headings(image, thresh, show_steps)
        
        # Crop the image to remove everything above the answer area
        height, width = thresh.shape
        # Clamp crop bounds to avoid empty slices
        answer_area_top = max(0, min(height - 2, int(answer_start_y + 50)))
        answer_area_bottom = max(answer_area_top + 2, min(height, int(height - 50)))

        # Crop the image to focus only on the answer area
        cropped_image = image[answer_area_top:answer_area_bottom, :]
        cropped_thresh = thresh[answer_area_top:answer_area_bottom, :]

        # Fallback: if cropping resulted in empty images, use full image threshold
        if cropped_thresh is None or cropped_thresh.size == 0:
            gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            cropped_thresh = cv2.threshold(gray_full, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            cropped_image = image.copy()
        
        if show_steps:
            cv2.imshow("Step 1c: Cropped Answer Area (No Top Part)", cv2.resize(cropped_image, (600, 600)))
            cv2.waitKey(0)
            cv2.imshow("Step 1d: Cropped Threshold", cv2.resize(cropped_thresh, (600, 600)))
            cv2.waitKey(0)
        
        # Create a simple rectangular contour for the answer zone (now starts at 0,0)
        answer_zone_contour = np.array([[
            [0, 0],
            [cropped_thresh.shape[1], 0],
            [cropped_thresh.shape[1], cropped_thresh.shape[0]],
            [0, cropped_thresh.shape[0]]
        ]], dtype=np.int32)
        
        return cropped_thresh, answer_zone_contour

    def detect_subject_headings(self, image: np.ndarray, thresh: np.ndarray, show_steps: bool = False) -> int:
        """
        Detect subject headings (Python, SQL, Power BI, Adv Stats) to find answer area start
        """
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Use adaptive threshold for text detection
        text_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
        
        if show_steps:
            cv2.imshow("Step 1b1: Text Detection Threshold", cv2.resize(text_thresh, (600, 800)))
            cv2.waitKey(0)
        
        # Find contours for text detection
        cnts, _ = cv2.findContours(text_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for text regions that could be subject headings
        heading_candidates = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            
            # Filter for text-like regions (horizontal rectangles)
            if (h > 15 and h < 50 and  # Height range for text
                w > 30 and w < 200 and  # Width range for text
                w > h * 2):  # Text is typically wider than tall
                heading_candidates.append((x, y, w, h))
        
        # Sort by y-coordinate to find the row with headings
        heading_candidates.sort(key=lambda x: x[1])
        
        if show_steps:
            debug_img = image.copy()
            for i, (x, y, w, h) in enumerate(heading_candidates[:20]):  # Show first 20 candidates
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(debug_img, f"H{i}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("Step 1b2: Heading Candidates (Green)", cv2.resize(debug_img, (600, 800)))
            cv2.waitKey(0)
        
        # Find the row with the most heading candidates (likely the subject row)
        if heading_candidates:
            # Group candidates by y-coordinate (same row)
            y_groups = {}
            for x, y, w, h in heading_candidates:
                y_key = y // 10 * 10  # Group by 10-pixel y ranges
                if y_key not in y_groups:
                    y_groups[y_key] = []
                y_groups[y_key].append((x, y, w, h))
            
            # Find the row with the most candidates (likely subject headings)
            best_row_y = min(y_groups.keys())
            max_candidates = 0
            for y_key, candidates in y_groups.items():
                if len(candidates) > max_candidates:
                    max_candidates = len(candidates)
                    best_row_y = y_key
            
            # Return the y-coordinate of the best row
            answer_start_y = best_row_y
            print(f"Step 1: Detected subject headings at y={answer_start_y} with {max_candidates} candidates")
            
            if show_steps:
                debug_img = image.copy()
                for x, y, w, h in y_groups[best_row_y]:
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.line(debug_img, (0, answer_start_y), (image.shape[1], answer_start_y), (255, 0, 0), 3)
                cv2.putText(debug_img, "Subject Headings Detected", (10, answer_start_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow("Step 1b3: Detected Subject Headings (Red)", cv2.resize(debug_img, (600, 800)))
                cv2.waitKey(0)
            
            return answer_start_y
        else:
            # Fallback: use a percentage-based approach
            print("Step 1: No headings detected, using fallback method")
            return int(image.shape[0] * 0.15)  # 15% from top

    def detect_bubbles(self, thresh: np.ndarray, image: np.ndarray, show_steps: bool = False) -> List[np.ndarray]:
        """
        Step 2: Detect individual bubbles using the already cleaned and cropped image
        """
        if show_steps:
            cv2.imshow("Step 2a: Cleaned and Cropped Image (Bubble Area Only)", cv2.resize(image, (600, 800)))
            cv2.waitKey(0)
            cv2.imshow("Step 2b: Cleaned Threshold Image", cv2.resize(thresh, (600, 800)))
            cv2.waitKey(0)
        
        # Use the already cleaned threshold image directly - no need to reprocess
        # The thresh parameter is already the cleaned, cropped threshold image
        
        # Apply minimal morphological operations to clean up any remaining noise
        if thresh is None or thresh.size == 0:
            # Fallback to threshold computed from the provided image
            if image is None or image.size == 0:
                raise ValueError("detect_bubbles received empty threshold and image")
            gray_fallback = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            thresh = cv2.threshold(gray_fallback, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        if show_steps:
            cv2.imshow("Step 2c: Final Cleaned Threshold", cv2.resize(cleaned, (600, 800)))
            cv2.waitKey(0)
        
        # Find contours for bubble detection
        cnts, _ = cv2.findContours(cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bubble_contours = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(c)
            
            # Filter for OMR bubbles - optimized for cleaned image
            if (0.8 <= aspect_ratio <= 1.2 and  # Circular shape
                12 < w < 45 and 12 < h < 45 and  # Size range for bubbles
                100 < area < 1200):  # Area range for bubbles
                
                # Additional checks to filter out any remaining noise
                perimeter = cv2.arcLength(c, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Check if it's circular enough (bubbles are more circular than noise)
                    if circularity > 0.6:  # Circular shapes
                        
                        # Additional check: ensure it's not too elongated
                        if w / h < 1.4 and h / w < 1.4:  # Not too elongated
                            bubble_contours.append(c)
        
        print(f"Step 2: Found {len(bubble_contours)} bubbles in cleaned image")
        
        if show_steps:
            debug_img = image.copy()
            cv2.drawContours(debug_img, bubble_contours, -1, (0, 0, 255), 2)
            cv2.imshow("Step 2d: Detected Bubbles (Red) - Using Cleaned Image", cv2.resize(debug_img, (600, 800)))
            cv2.waitKey(0)
        
        return bubble_contours

    def sort_bubbles_by_columns(self, bubble_contours: List[np.ndarray], image: np.ndarray, 
                               show_steps: bool = False) -> List[List[np.ndarray]]:
        """
        Step 3: Sort bubbles into straight 4-column alignment (A, B, C, D)
        """
        if not bubble_contours:
            return []
        
        print(f"Step 3: Organizing {len(bubble_contours)} bubbles into 4-column layout")
        
        # First, sort all bubbles by y-coordinate to get rows
        bubble_contours.sort(key=lambda c: cv2.boundingRect(c)[1])
        
        # Group bubbles by rows (questions)
        rows = []
        current_row = []
        last_y = None
        
        for bubble in bubble_contours:
            (x, y, w, h) = cv2.boundingRect(bubble)
            current_y = y + h // 2  # Use center y-coordinate
            
            if last_y is None or abs(current_y - last_y) < 20:  # Same row if within 20 pixels
                current_row.append(bubble)
            else:
                if current_row:
                    rows.append(current_row)
                current_row = [bubble]
            
            last_y = current_y
        
        if current_row:
            rows.append(current_row)
        
        print(f"   Found {len(rows)} rows of bubbles")
        
        # Now organize each row into 4 columns (A, B, C, D)
        questions = []
        for row_idx, row_bubbles in enumerate(rows):
            if len(row_bubbles) >= 4:  # Need at least 4 bubbles per row
                # Sort bubbles in this row by x-coordinate
                row_bubbles.sort(key=lambda c: cv2.boundingRect(c)[0])
                
                # Group into sets of 4 (A, B, C, D)
                for i in range(0, len(row_bubbles), 4):
                    if i + 3 < len(row_bubbles):
                        question_bubbles = row_bubbles[i:i+4]
                        questions.append(question_bubbles)
        
        print(f"   Organized into {len(questions)} questions with 4 options each")
        
        if show_steps:
            debug_img = image.copy()
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # B, G, R, C
            labels = ['A', 'B', 'C', 'D']
            
            # Show first 10 questions with proper A, B, C, D labeling
            for i, question in enumerate(questions[:10]):
                for j, bubble in enumerate(question):
                    (x, y, w, h) = cv2.boundingRect(bubble)
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), colors[j], 2)
                    cv2.putText(debug_img, f"Q{i+1}{labels[j]}", (x, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[j], 1)
            
            cv2.imshow("Step 3: 4-Column Alignment (A, B, C, D)", cv2.resize(debug_img, (600, 800)))
            cv2.waitKey(0)
        
        return questions

    def grade_answers(self, questions: List[List[np.ndarray]], thresh: np.ndarray, 
                     set_name: str, show_steps: bool = False) -> Dict[str, Any]:
        """
        Step 4: Match detected answers with answer key and calculate accuracy
        """
        key_to_use = self.answer_key.get(set_name, {})
        if not key_to_use:
            raise Exception(f"No answer key found for set: {set_name}")
        
        subject_scores = {s: 0 for s in self.SUBJECTS}
        total_correct = 0
        detailed_results = []
        
        print(f"Step 4: Grading {len(questions)} questions against answer key")
        
        for i, question_bubbles in enumerate(questions):
            question_num_str = str(i + 1)
            is_correct = False
            student_answer = "No Answer"
            
            # Calculate pixel density for each bubble (A, B, C, D)
            pixel_counts = []
            for bubble in question_bubbles:
                # Create mask for this specific bubble
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [bubble], -1, 255, -1)
                
                # Count non-zero pixels within the bubble
                filled_pixels = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
                pixel_counts.append(filled_pixels)
            
            # Determine which bubble is filled (highest pixel count)
            if max(pixel_counts) > 100:  # Threshold for filled bubble
                student_idx = np.argmax(pixel_counts)
                student_answer = chr(65 + student_idx)  # A, B, C, D
                
                # Check against answer key
                correct_char = key_to_use.get(question_num_str, "").lower()
                correct_idx = self.ANSWER_MAP.get(correct_char, -1)
                
                if student_idx == correct_idx:
                    is_correct = True
                    total_correct += 1
                    
                    # Update subject scores
                    for subject, (start, end) in self.SUBJECTS.items():
                        if start <= i + 1 <= end:
                            subject_scores[subject] += 1
                            break
            
            detailed_results.append({
                "question": question_num_str,
                "student_answer": student_answer,
                "correct_answer": key_to_use.get(question_num_str, "N/A"),
                "is_correct": is_correct,
                "pixel_counts": pixel_counts
            })
        
        # Calculate accuracy
        total_questions = len(questions)
        accuracy = (total_correct / total_questions) * 100 if total_questions > 0 else 0
        
        results = {
            "total_score": total_correct,
            "total_questions": total_questions,
            "accuracy": accuracy,
            "subject_scores": subject_scores,
            "detailed_results": detailed_results
        }
        
        return results

    def grade(self, image: np.ndarray, set_name: str, debug: bool = False) -> Dict[str, Any]:
        """
        Main grading function that orchestrates all steps
        """
        print("=== OMR Grading Process Started ===")
        
        # Step 1: Detect bubble area and get cleaned image
        thresh, answer_zone = self.detect_bubble_area(image, show_steps=debug)
        
        # Get the cleaned and cropped image (bubble area only)
        # This is the image that was cropped in detect_bubble_area
        height, width = image.shape[:2]
        answer_start_y = self.detect_subject_headings(image, thresh, show_steps=False)
        # Clamp cleaned crop
        top = max(0, min(height - 2, int(answer_start_y + 50)))
        bottom = max(top + 2, min(height, int(height - 50)))
        cleaned_image = image[top:bottom, :]
        if cleaned_image is None or cleaned_image.size == 0:
            cleaned_image = image.copy()
        
        # Step 2: Detect individual bubbles using the cleaned image
        bubble_contours = self.detect_bubbles(thresh, cleaned_image, show_steps=debug)
        
        # Validate bubble count
        expected_bubbles = 400  # 100 questions * 4 options
        if not (expected_bubbles - 20 <= len(bubble_contours) <= expected_bubbles + 20):
            print(f"Warning: Expected ~{expected_bubbles} bubbles, found {len(bubble_contours)}")
        
        # Step 3: Sort bubbles by columns and questions using cleaned image
        questions = self.sort_bubbles_by_columns(bubble_contours, cleaned_image, show_steps=debug)
        
        # Step 4: Grade answers
        results = self.grade_answers(questions, thresh, set_name, show_steps=debug)
        
        print("=== OMR Grading Process Completed ===")
        
        if debug:
            cv2.destroyAllWindows()
        
        return results