# Wound Analysis Flask Application

## Overview
This Flask application analyzes wound images uploaded by users. It processes the images to detect wound areas and generates segmentation masks, storing results in a MongoDB database. The application provides a user-friendly interface for uploading images and viewing analysis results.

## Features
- Upload images of wounds for analysis.
- Image processing to detect and segment wounds.
- Store results in MongoDB for future reference.
- View historical analysis results on a dedicated page.

## Prerequisites
- Python 3.x
- MongoDB
- Cloudinary account (for image storage)
- Required Python packages (listed in `requirements.txt`)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ashwanth-23/Wound-Detect.git
   cd Wound-Detect
2. Set up a virtual environment (optional but recommended):
   python -m venv venv
  source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. Install the required packages:
   pip install -r requirements.txt
4. Set up MongoDB:
  . Ensure that MongoDB is installed and running.
  . Create a database named wound_analysis and a collection named wound_records in your MongoDB instance.
5. Configure Cloudinary:

  .  Create a Cloudinary account if you don't have one.
  . Update the Cloudinary credentials in app.py with your cloud_name, api_key, and api_secret.

6. Run the application:
   python app.py

Usage
Home Page:
Upload an image of a wound using the upload button.
Click the "Analyze Wound" button to process the image.

Results Page:
View the original image and the processed image with detected wound contours.
The calculated wound area will be displayed alongside the images.

History Page:
Access previously analyzed wounds and view their details stored in the database.

Folder Structure:
Wound-Detect/
│
├── app.py                     # Main Flask application for wound analysis
├── requirements.txt           # Python package dependencies
├── templates/                 # HTML templates for the application
│   ├── index.html             # Home page for image upload
│   ├── result.html            # Results page showing analysis
│   └── history.html           # Page for viewing past analyses
└── images/                    # Folder for sample images (to be added)


License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
.Flask for building the web application.
.OpenCV for image processing.
.Cloudinary for image storage and management.
.MongoDB for data storage.
Contact
For inquiries, please reach out to Ashwanth at Bakkannaashwanth@gmail.com.




   
 






