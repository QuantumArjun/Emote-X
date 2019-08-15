import soundfile
import librosa
import numpy as np
import pickle


AVAILABLE_EMOTIONS = {
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fear",
    "disgust",
    "ps", # pleasant surprised
    "boredom"
}


def get_label(audio_config):
    """Returns label corresponding to which features are to be extracted
        e.g:
    audio_config = {'mfcc': True, 'chroma': True, 'contrast': False, 'tonnetz': False, 'mel': False}
    get_label(audio_config): 'mfcc-chroma'
    """
    features = ["mfcc", "chroma", "mel", "contrast", "tonnetz"]
    label = ""
    for feature in features:
        if audio_config[feature]:
            label += f"{feature}-"
    return label.rstrip("-")


def get_dropout_str(dropout, n_layers=3):
    if isinstance(dropout, list):
        return "_".join([ str(d) for d in dropout])
    elif isinstance(dropout, float):
        return "_".join([ str(dropout) for i in range(n_layers) ])


def get_first_letters(emotions):
    return "".join(sorted([ e[0].upper() for e in emotions ]))


def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
            - Pitch_cepstrum
            - Formants
            - RSME
            - Poly_features
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    pitch_cepstrum = kwargs.get('pitch_cepstrum')
    formants = kwargs.get('formants')
    rmse = kwargs.get('rmse')
    chroma_cens = kwargs.get('chroma_cens')
    poly_features = kwargs.get('poly_features')
    deriv = kwargs.get('deriv')
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)     
            result = np.hstack((result, np.mean(mfccs.T, axis=0)))
            #deriv
            if deriv:
                deriv_mfcc = librosa.feature.delta(mfccs)
                result = np.hstack((result, np.mean(deriv_mfcc.T, axis=0)))
        if chroma:
            chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
            result = np.hstack((result, np.mean(chroma.T, axis=0)))
            #deriv
            if deriv:
                deriv_chroma = librosa.feature.delta(chroma)
                result = np.hstack((result, np.mean(deriv_chroma.T, axis=0)))
        if mel:
            mel = librosa.feature.melspectrogram(X, sr=sample_rate)
            result = np.hstack((result, np.mean(mel.T, axis=0)))
            #deriv
            if deriv:
                deriv_mel = librosa.feature.delta(mel)
                result = np.hstack((result, np.mean(deriv_mel.T, axis=0)))
        if contrast:
            contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
            result = np.hstack((result, np.mean(contrast.T, axis = 0)))
            #deriv
            if deriv:
                deriv_contrast = librosa.feature.delta(contrast)
                result = np.hstack((result, np.mean(deriv_contrast.T, axis=0)))
        if tonnetz:
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate)
            result = np.hstack((result, np.mean(tonnetz.T, axis=0)))
            #deriv
            if deriv:
                deriv_tonnetz = librosa.feature.delta(tonnetz)
                result = np.hstack((result, np.mean(deriv_tonnetz.T, axis=0))) 

        if rmse:
            S, phase = librosa.magphase(librosa.stft(X))
            rmse = librosa.feature.rmse(S=S)
            result = np.hstack((result, np.mean(rmse.T, axis=0)))
            #deriv
            if deriv:
                deriv_rmse = librosa.feature.delta(rmse)
                result = np.hstack((result, np.mean(deriv_rmse.T, axis=0))) 

        if chroma_cens:
            chroma_cens = librosa.feature.chroma_cens(y=X, sr=sample_rate)
            result = np.hstack((result, np.mean(chroma_cens.T, axis=0)))
            #deriv
            if deriv:
                deriv_chroma_cens = librosa.feature.delta(chroma_cens)
                result = np.hstack((result, np.mean(deriv_chroma_cens.T, axis=0)))

        if poly_features:
            if not rsme:
                S, phase = librosa.magphase(librosa.stft(X))
            poly_features = librosa.feature.poly_features(y=S, sr=sample_rate, order=2)
            result = np.hstack((result, np.mean(poly_features.T, axis=0)))
            #deriv
            if deriv:
                deriv_poly_features = librosa.feature.delta(poly_features)
                result = np.hstack((result, np.mean(deriv_poly_features.T, axis=0)))
                         
    return result


def get_best_estimators(classification):
    """
    Loads the estimators that are pickled in `grid` folder
    Note that if you want to use different or more estimators,
    you can fine tune the parameters in `grid_search.py` script
    and run it again ( may take hours )
    """
    if classification:
        return pickle.load(open("grid/best_classifiers.pickle", "rb"))
    else:
        return pickle.load(open("grid/best_regressors.pickle", "rb"))


def get_audio_config(features_list):
    """
    Converts a list of features into a dictionary understandable by
    `data_extractor.AudioExtractor` class
    """
    audio_config = {'mfcc': False, 'chroma': False, 'mel': False, 'contrast': False, 'tonnetz': False, 
                   'rmse': False, 'poly_feature':False,'chroma_cens':False, 'deriv':False}
    for feature in features_list:
        if feature not in audio_config:
            raise TypeError(f"Feature passed: {feature} is not recognized.")
        audio_config[feature] = True
    return audio_config
    