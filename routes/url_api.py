import base64
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import io
import re
import os
from utils import load_image_from_source, validate_image, encode_image, compress_image
from openai import OpenAI
from testMap_utils import map_test_code, map_ref_code
from config import load_config

# Setup logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration values
config = load_config()
API_KEY = config['api_key']
MODEL = config['model']
MAX_SIZE_MB = config['max_size_mb']
threshold = config['threshold']
client = OpenAI(api_key=API_KEY)

router = APIRouter()

# Pydantic model for input
class ImageURL(BaseModel):
    url: str

@router.post("/extract_and_map_tests_url/")
async def extract_and_map_tests(image_url: ImageURL):
    """
    This endpoint extracts test names from an image given via an HTTP/HTTPS or local file path URL
    and maps them to test codes.
    """
    try:
        # Log the start of the request
        logger.info(f"Received request for image processing from URL: {image_url.url}")

        # Check if the URL is a local path (starts with localhost)
        local_path_pattern = r'http://127\.0\.0\.1(:\d+)?/["\']?(.*)["\']?'
        if re.match(local_path_pattern, image_url.url):
            local_path_match = re.match(local_path_pattern, image_url.url)
            if local_path_match:
                # Extract the local path and replace forward slashes with backslashes for Windows compatibility
                local_path = local_path_match.group(2)
                local_path = local_path.replace('/', '\\')  # Ensure consistent backslashes
                local_path = os.path.normpath(local_path)   # Normalize the path for OS compatibility
                logger.info(f"Interpreted local file path: {local_path}")

                # Check if the file exists before proceeding
                if not os.path.exists(local_path):
                    logger.error(f"File not found at path: {local_path}")
                    raise HTTPException(status_code=404, detail=f"File not found: {local_path}")

                image = load_image_from_source(local_path)
            else:
                raise HTTPException(status_code=400, detail="Invalid local path URL format.")
        else:
            # Handle external HTTP/HTTPS URL
            image = load_image_from_source(image_url.url)

        # Step 1: Convert image to bytes and validate
        image_bytes = io.BytesIO()
        image.save(image_bytes, format=image.format)
        image_bytes.seek(0)
        validate_image(image_bytes)

        # Step 2: Compress the image if needed
        compressed_image_path = compress_image(image_bytes, MAX_SIZE_MB)
        logger.info(f"Image compressed and saved at: {compressed_image_path}")

        # Step 3: Encode the image to Base64
        with open(compressed_image_path, 'rb') as f:
            encoded_image = encode_image(f.read())
        logger.info("Image encoded to Base64 successfully.")

        # Step 4: Read the prompt template
        logger.info("Reading the prompt template for OpenAI request.")
        with open('prompt_template.txt', 'r') as prompt_file:
            prompt_template = prompt_file.read()

        # Step 5: Send the image and prompt to OpenAI's API for data extraction
        logger.info("Sending the image to OpenAI API for test name extraction.")
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that responds in JSON format. Help me to get the patient's data and prescribed pathological lab tests extracted from the prescription given by a doctor, hospital, or lab."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_template
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                        }
                    ]
                }
            ],
            temperature=0.0,
        )

        # Parse the GPT response and extract test names
        logger.debug("Parsing the response from OpenAI API.")
        gpt_response = response.choices[0].message.content.split('```json')[-1].split('```')[0]
        extracted_data = json.loads(gpt_response)

        # Extract the prescribed test names
        test_names = extracted_data.get("prescribed_test", [])
        logger.info(f"Extracted {len(test_names)} test names from the image.")

        if not test_names:
            logger.warning("No test names found in the extracted data.")
            raise HTTPException(status_code=400, detail="No test names found in the extracted data.")

        # Step 5.1: Map test names to test codes
        logger.info("Mapping extracted test names to test codes.")
        mapped_tests = []
        for input_test_name in test_names:
            matched_test_name, matched_test_code = map_test_code(input_test_name, threshold)
            if (matched_test_name, matched_test_code) != (None, None):
                logger.debug(f"Mapped test name: {input_test_name} -> {matched_test_name} (Code: {matched_test_code})")
                mapped_tests.append({
                    "input_test_name": input_test_name,
                    "matched_test_name": matched_test_name,
                    "matched_test_code": matched_test_code
                })

        # Step 5.2: Map ref names to ref codes
        ref_name = extracted_data['referrer_name']
        logger.info(f"Mapping ref names to ref code for --> {ref_name}")
        matched_ref_name, matched_ref_code, matched_ref_type = map_ref_code(ref_name, threshold)
        logger.debug(f"Mapped ref name: {matched_ref_name} -> Mapped Ref Code: {matched_ref_code} -> Mapped Ref type: {matched_ref_type}")
        extracted_data['matched_ref_name'] = matched_ref_name
        extracted_data['matched_ref_code'] = matched_ref_code
        extracted_data['matched_ref_type'] = matched_ref_type

        # Step 6: Combine the extracted data with mapped test codes
        logger.info("Combining extracted data with mapped test codes.")
        combined_result = {
            "extracted_data": extracted_data,
            "mapped_tests": mapped_tests
        }

        # Return the response
        logger.info("Successfully processed the request. Returning the results.")
        return JSONResponse(content={
            "extracted_data": combined_result["extracted_data"],
            "mapped_tests": combined_result["mapped_tests"]
        })

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during processing: {e}")
