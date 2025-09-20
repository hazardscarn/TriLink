import pandas as pd
import base64
import mimetypes
import os
from google import genai
from google.genai import types
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()



def save_binary_file(file_name, data):
    """Save binary data to a file"""
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"File saved to: {file_name}")

def generate_image_from_prompt(prompt, ticket_id, output_dir="../data/images"):
    """
    Generate an image from a text prompt using Gemini and save it with ticket_id as filename
    """
    try:
        # Initialize Gemini client
        client = genai.Client(
            api_key=os.environ.get("GOOGLE_API_KEY"),
        )
        
        model = "gemini-2.5-flash-image-preview"
        
        # Create content with the image generation prompt
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]
        
        # Configure for image generation
        generate_content_config = types.GenerateContentConfig(
            response_modalities=[
                "IMAGE",
                "TEXT",
            ],
        )
        
        # Generate content
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if (
                chunk.candidates is None
                or chunk.candidates[0].content is None
                or chunk.candidates[0].content.parts is None
            ):
                continue
            
            # Check if we have image data
            if (chunk.candidates[0].content.parts[0].inline_data and 
                chunk.candidates[0].content.parts[0].inline_data.data):
                
                inline_data = chunk.candidates[0].content.parts[0].inline_data
                data_buffer = inline_data.data
                file_extension = mimetypes.guess_extension(inline_data.mime_type)
                
                # Use ticket_id as filename, force .jpeg extension as requested
                file_name = f"{ticket_id}.jpeg"
                file_path = os.path.join(output_dir, file_name)
                
                save_binary_file(file_path, data_buffer)
                return file_path
            else:
                # Print any text responses
                if hasattr(chunk, 'text') and chunk.text:
                    print(f"Text response for ticket {ticket_id}: {chunk.text}")
        
        return None
        
    except Exception as e:
        print(f"Error generating image for ticket {ticket_id}: {str(e)}")
        return None

def process_dataframe_images(df, output_dir="../data/images"):
    """
    Process dataframe and generate images for each row
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if required columns exist
    required_columns = ['ticket_id', 'image_generation_prompt']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Process each row
    results = []
    total_rows = len(df)
    
    for index, row in df.iterrows():
        ticket_id = row['ticket_id']
        prompt = row['image_generation_prompt']
        
        print(f"Processing ticket {ticket_id} ({index + 1}/{total_rows})...")
        print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        
        # Skip if prompt is empty or NaN
        if pd.isna(prompt) or str(prompt).strip() == '':
            print(f"Skipping ticket {ticket_id}: Empty prompt")
            results.append({
                'ticket_id': ticket_id,
                'status': 'skipped',
                'reason': 'empty_prompt',
                'file_path': None
            })
            continue
        
        # Generate image
        file_path = generate_image_from_prompt(prompt, ticket_id, output_dir)
        
        if file_path:
            results.append({
                'ticket_id': ticket_id,
                'status': 'success',
                'reason': None,
                'file_path': file_path
            })
            print(f"✓ Successfully generated image for ticket {ticket_id}")
        else:
            results.append({
                'ticket_id': ticket_id,
                'status': 'failed',
                'reason': 'generation_failed',
                'file_path': None
            })
            print(f"✗ Failed to generate image for ticket {ticket_id}")
        
        print("-" * 50)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Print summary
    success_count = len(results_df[results_df['status'] == 'success'])
    failed_count = len(results_df[results_df['status'] == 'failed'])
    skipped_count = len(results_df[results_df['status'] == 'skipped'])
    
    print(f"\nSUMMARY:")
    print(f"Total tickets processed: {total_rows}")
    print(f"Successfully generated: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped: {skipped_count}")
    
    return results_df

# Example usage
if __name__ == "__main__":
    # Make sure you have set your GEMINI_API_KEY environment variable
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: Please set the GEMINI_API_KEY environment variable")
        exit(1)
    
    # Load your dataframe (replace with your actual data loading code)
    # df = pd.read_csv("your_data.csv")  # or however you load your dataframe
    
    # Example dataframe for testing
    sample_data = {
        'ticket_id': ['T001', 'T002', 'T003'],
        'customer_id': ['C001', 'C002', 'C003'],
        'technician_id': ['TECH01', 'TECH02', 'TECH01'],
        'issue_description': ['Network issue', 'Software bug', 'Hardware failure'],
        'image_generation_prompt': [
            'A technical diagram showing network connectivity issues',
            'A screenshot of software error with red warning messages',
            'A photo of damaged computer hardware components'
        ]
    }
    df = pd.DataFrame(sample_data)
    
    # Process the dataframe
    results = process_dataframe_images(df, output_dir="../data/images")
    
    # Optionally save results to CSV
    results.to_csv("image_generation_results.csv", index=False)
    print(f"\nResults saved to: image_generation_results.csv")