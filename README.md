# Osu Map Generator
by Edan Meyer, Yazeed Mahmoud ... add names here

### How To Use
1. Download the formatted dataset, decompress it, and put the contents in the `data/` directory. This should leave you with a `data/formatted_beatmaps/` directory that contains the song and beatmap data.
2. (Optional) Create a new Python environment using software of your choice. Python version 3.9.12 is recommended, and version 3.8+ is required.
3. Download the requirements with `pip install -r requirements.txt`.
4. From the root directory (this is important), run `python preprocessing/create_vocab.py`. This will take a few minutes, and will create the vocabulary used for text preprocessing.
5. After the previous command finishes running, run `python training.py` to train a model.
6. There is currently no way to produce sample maps.

### Audio preprocessing
1. **Song file sementation**: first, we check if a segmented version of our song already exists, if not, we break it down into equal sized chunks and store them in a directory with the song's name
2. **Audio conversion to Mono**: combine the stereo channels to create a single-channelled monotune signal
3. **Apply STFT**: to convert to audio signals from time domain to frequency domain
4. **Apply MEL-filtering**: to compress the dimensionality of the produced spectra down to 80 frequency bands
5. **Apply numerically-stable log-scaling**: to give rise to frequency bands that represent actual human loudness perception levels
6. **Sample retrieval**: save all the processed segments into one large numpy array, for each hit-object, retrieve the segment at the corresponding time snapshot, in addition to a few antecedent and subsequent segments.*
