import os
import glob
from pydub import AudioSegment

def split_single_audio(audio_path, file_path, save_dir):
    """ Split audio in 5 seconds chunks and saves them in the save_dir 
    """
    path = audio_path + file_path
    save_dir = save_dir + file_path.split("/")[0]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    # Split audio in 5 seconds chunks
    split_audio = AudioSegment.from_file(path, 'au')
    duration = 5000
    
    for i in range(len(split_audio) // duration):
        chunk = split_audio[i*duration:(i+1)*duration]
        chunk.export(save_dir + "/" + file_path.split("/")[0] + '.' + 
                     file_path.split(".")[1] + '_' + str(i) + ".wav", format="wav")

    
def split():
    audio_path = 'data/original_audio/'
    file_paths = glob.glob(audio_path + '**/*.au', recursive=True)
    for file_path in file_paths:
        pth = file_path.split("\\")[1:]
        file_path = pth[0] + "/" + pth[1]
        save_path = 'data/split_audio/'
        split_single_audio(audio_path, file_path, save_path)
    
