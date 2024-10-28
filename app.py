from flask import Flask, render_template, request, redirect, url_for, flash, session
from datetime import datetime
from pymongo import MongoClient
import cv2
import numpy as np
import pytz
import cloudinary
import cloudinary.uploader
import os
from bson import ObjectId
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tempfile
from pathlib import Path

app = Flask(__name__)
app.secret_key = 'FLASK_SECRET_KEY'  # Change this in production

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['wound_analysis']
records = db['wound_records']

# Cloudinary configuration
cloudinary.config(
    cloud_name="dnhonxbwe",
    api_key="666632788637786",
    api_secret="AqIpmmBaDcMLZ3Ee-mLqLo6F6og"
)

# Create a temporary directory for image processing
TEMP_DIR = Path(tempfile.gettempdir()) / 'wound_analysis'
TEMP_DIR.mkdir(exist_ok=True)

def ensure_directory(path):
    """Ensure directory exists, create if it doesn't."""
    path.mkdir(parents=True, exist_ok=True)
    return path

def safe_imread(image_path):
    """Safely read an image file with error handling."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to read image at {image_path}. File may be corrupted or in an unsupported format.")
    
    return img

def process_wound_image(image_path):
    """
    Process wound image with precise calibrated measurements.
    Returns: processed image path, area, and segmentation mask.
    """
    try:
        # Read the image with error handling
        img = safe_imread(image_path)

        # Convert to multiple color spaces for better analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        

        # Extract individual channels
        _, a, _ = cv2.split(lab)

        # Enhanced red detection in HSV space with precise bounds
        lower_red1 = np.array([0, 120, 100])
        upper_red1 = np.array([8, 255, 255])
        lower_red2 = np.array([172, 120, 100])
        upper_red2 = np.array([180, 255, 255])

        # Create initial masks
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask_red1, mask_red2)

        # Enhanced a* channel processing
        a_blur = cv2.GaussianBlur(a, (5, 5), 0)
        _, a_mask = cv2.threshold(a_blur, 135, 255, cv2.THRESH_BINARY)

        # Combine masks
        combined_mask = cv2.bitwise_and(red_mask, a_mask)

        # Refined morphological operations
        kernel_small = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_small)

        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            raise ValueError("No wound detected in the image.")

        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Precise contour smoothing
        epsilon = 0.002 * cv2.arcLength(largest_contour, True)
        smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Draw the result
        result = img.copy()
        cv2.drawContours(result, [smoothed_contour], -1, (0, 255, 255), 2)

        # Calculate area using the precise conversion factor
        pixel_area = cv2.contourArea(smoothed_contour)
        cm2_per_pixel = 0.0264583333  # Exact conversion factor
        area_cm2 = pixel_area * cm2_per_pixel

        # Save processed image to temporary directory
        processed_path = TEMP_DIR / f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(processed_path), result)

        if not processed_path.exists():
            raise IOError(f"Failed to save processed image to {processed_path}")

        return processed_path, round(area_cm2, 2), combined_mask

    except Exception as e:
        raise Exception(f"Error processing wound image: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        flash('No image file provided')
        return redirect(url_for('home'))

    file = request.files['image']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('home'))

    try:
        # Create temporary file with proper extension
        temp_file = TEMP_DIR / f"original_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        file.save(str(temp_file))

        if not temp_file.exists():
            raise FileNotFoundError(f"Failed to save uploaded file to {temp_file}")

        # Upload original to Cloudinary
        original_upload = cloudinary.uploader.upload(str(temp_file))
        original_url = original_upload['secure_url']

        # Process image
        processed_path, area, mask = process_wound_image(temp_file)

        # Upload processed image to Cloudinary
        processed_upload = cloudinary.uploader.upload(str(processed_path))
        processed_url = processed_upload['secure_url']

        # Define the Indian timezone
        india_timezone = pytz.timezone("Asia/Kolkata")

        utc_time = datetime.utcnow()
        india_time = utc_time.replace(tzinfo=pytz.utc).astimezone(india_timezone)

        india_time_naive = india_time.replace(tzinfo=None)



        # Save to MongoDB
        record = {
            'original_image': original_url,
            'processed_image': processed_url,
            'area_cm2': area,
            'timestamp': india_time_naive
        }
        record_id = records.insert_one(record).inserted_id

        # Store results in session for the result page
        session['last_analysis'] = {
            'original_image': original_url,
            'processed_image': processed_url,
            'area_cm2': area,
            'record_id': str(record_id)
        }

        # Generate visualization
        visualize_results(temp_file, processed_path, mask, area)

        # Clean up temporary files
        temp_file.unlink(missing_ok=True)
        processed_path.unlink(missing_ok=True)

        # Redirect to result page instead of history
        return redirect(url_for('result'))

    except Exception as e:
        flash(f'Error processing image: {str(e)}')
        return redirect(url_for('home'))
    finally:
        # Ensure cleanup of temporary files even if an error occurs
        for temp_file in TEMP_DIR.glob("*"):
            try:
                temp_file.unlink(missing_ok=True)
            except Exception:
                pass

@app.route('/result')
def result():
    try:
        # Get analysis results from session
        analysis_data = session.get('last_analysis')
        if not analysis_data:
            flash('No analysis results found')
            return redirect(url_for('home'))

        # Clear the session data after retrieving it
        session.pop('last_analysis', None)

        # Render the result template with the analysis data
        return render_template('result.html',
                             original_image=analysis_data['original_image'],
                             processed_image=analysis_data['processed_image'],
                             area_cm2=analysis_data['area_cm2'])

    except Exception as e:
        flash(f'Error displaying results: {str(e)}')
        return redirect(url_for('home'))

def visualize_results(original_img_path, processed_img_path, mask, area):
    """
    Visualize results with precise area measurement and save to temporary directory.
    """
    try:
        original_img = safe_imread(original_img_path)
        processed_img = safe_imread(processed_img_path)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        plt.imshow(original_rgb)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        plt.imshow(processed_rgb)
        plt.title(f'Detected Wound\nArea: {area} cm²')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(mask, cmap='gray')
        plt.title('Segmentation Mask')
        plt.axis('off')

        plt.tight_layout()
        
        # Save visualization to temporary directory
        vis_path = TEMP_DIR / f"visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(str(vis_path))
        plt.close()

    except Exception as e:
        raise Exception(f"Error visualizing results: {str(e)}")

@app.route('/history')
def history():
    try:
        all_records = list(records.find().sort('timestamp', -1))
         # Set the Indian timezone
        india_timezone = pytz.timezone("Asia/Kolkata")
        for record in all_records:
            record['timestamp'] = record['timestamp'].astimezone(india_timezone)
        return render_template('history.html', records=all_records)
    except Exception as e:
        flash(f'Error retrieving history: {str(e)}')
        return redirect(url_for('home'))

if __name__ == '__main__':
    # Ensure temporary directory exists
    ensure_directory(TEMP_DIR)
    app.run(debug=True)