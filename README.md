# Osu Map Generator
by Edan Meyer, ... add names here

### How To Use
1. Download the formatted dataset, decompress it, and put the contents in the `data/` directory. This should leave you with a `data/formatted_beatmaps/` directory that contains the song and beatmap data.
2. (Optional) Create a new Python environment using software of your choice. Python version 3.9.12 is recommended, and version 3.8+ is required.
3. Download the requirements with `pip install -r requirements.txt`.
4. From the root directory (this is important), run `python preprocessing/create_vocab.py`. This will take a few minutes, and will create the vocabulary used for text preprocessing.
5. After the previous command finishes running, run `python training.py` to train a model.
6. There is currently no way to produce sample maps.