import os
from datetime import datetime
import pyzipper

def audit_receipt(staff_id, full_name, image_paths, image_source, inference_time):
    current_date = datetime.now().strftime("%d-%b-%Y %H:%M")
    receipt = f"""
________                           ___.                            
\_____  \  ______    ____    ____  \_ |__  _____   _______   ____  
 /   |   \ \____ \ _/ __ \  /    \  | __ \ \__  \  \_  __ \_/ __ \ 
/    |    \|  |_> >\  ___/ |   |  \ | \_\ \ / __ \_ |  | \/\  ___/ 
\_______  /|   __/  \___  >|___|  / |___  /(____  / |__|    \___  >
        \/ |__|         \/      \/      \/      \/              \/ 

Audit Trail - Encrypted Package
Date: {current_date}
Staff ID: {staff_id}
Name: {full_name}
Images Packaged: {len(image_paths)}
Source: {image_source}
Inference Duration: {inference_time:.2f} seconds
Encryption: Enabled (AES-256)
"""

    return receipt

def encrypt_and_package_images(image_paths, staff_id, password, full_name="unknown", image_source="unknown", inference_time=0.0):
    try:
        output_dir = "inference_packaged"
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"{staff_id}_{timestamp}.zip"
        zip_path = os.path.join(output_dir, zip_filename)

        zip_password = f"{staff_id}{password}".encode("utf-8")
        
        # Call the external audit_receipt function to generate the audit text.
        audit_text = audit_receipt(staff_id, full_name, image_paths, image_source, inference_time)

        with pyzipper.AESZipFile(zip_path, 'w',
                                 compression=pyzipper.ZIP_DEFLATED,
                                 encryption=pyzipper.WZ_AES) as zipf:
            zipf.setpassword(zip_password)
            for path in image_paths:
                if os.path.exists(path):
                    arcname = os.path.basename(path)
                    zipf.write(path, arcname=arcname)
                else:
                    return f"Error: File not found - {path}"
            # Aaudit receipt into the zip archive... "audit.txt"
            zipf.writestr("audit.txt", audit_text)

        return f"Success: Encrypted package created - '{zip_path}'"
    except Exception as e:
        return f"Encryption failed: {str(e)}"
