import asyncio
import os
import subprocess
import logging

def convert_to_wav(file_path):
    """Convert audio file to WAV format (16kHz, mono)"""
    try:
        output_file_path = file_path.split('.')[0] + '_converted.wav'
        command = f'ffmpeg -i {file_path} -ac 1 -ar 16000 {output_file_path}'
        subprocess.run(command, shell=True, check=True)
        logging.info(f"Audio converted to WAV: {output_file_path}")
        return output_file_path
    except Exception as e:
        logging.error(f"Error converting audio: {e}")
        return None

def unique_filename(file: str, path: str) -> str:
    """Change file name if file already exists"""
    # check if file exists
    if not os.path.exists(os.path.join(path, file)):
        return file
    # get file name and extension
    filename, filext = os.path.splitext(file)
    # get full file path without extension only
    filexx = os.path.join(path, filename)
    # create incrementing variable
    i = 1
    # determine incremented filename
    while os.path.exists(f'{filexx}_{str(i)}{filext}'):
        # update the incrementing variable
        i += 1
    return f'{filename}_{str(i)}{filext}'

async def delete_file(file_path: str) -> None:
    """Delete file asynchronously"""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, os.remove, file_path)
        logging.info(f"File {file_path} successfully deleted")
    except FileNotFoundError:
        logging.info(f"File {file_path} not found")
    except OSError as e:
        logging.error(f"Error deleting file {file_path}: {e}")