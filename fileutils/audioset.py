import numpy
import struct
import subprocess
import librosa
from scipy.io import wavfile
from fileutils import smart_open

def readAudioSet(filename):
    """
    Reads audio and labels from a Google Audio Set file (disguised as .flac).
    Returns two variables:
      * wav -- a 2-D numpy float32 array, where each row is a waveform
        (10 seconds @ 16 kHz, mono);
      * labels -- a 2-D numpy int32 array of zeros and ones, where each row
        indicates the sound events active in the corresponding waveform.
    """
    wav, _ = librosa.core.load(filename, sr = 16000, dtype = "float32")
    with smart_open(filename, "rb") as f:
        f.seek(-12, 2)
        nClips, nSamples, nLabels = struct.unpack("<3i", f.read(12))
        wav = wav.reshape(nClips, nSamples)
        nBytes = (nLabels - 1) / 8 + 1
        f.seek(-12 - nClips * nBytes, 2)
        data = struct.unpack("<%dB" % (nClips * nBytes), f.read(nClips * nBytes))
        bytes = numpy.array(data).reshape(nClips, nBytes)
        labels = numpy.zeros((nClips, nLabels), dtype = "int32")
        for i in xrange(nLabels):
            labels[:,i] = (bytes[:, i / 8] >> (i % 8)) & 1
        return wav, labels

def writeAudioSet(filename, wav, labels):
    """
    Writes audio and labels to a Google Audio Set file (disguised as .flac).
    Takes two variables as input:
      * wav -- a 2-D numpy array, where each row is a waveform
        (10 seconds @ 16 kHz, mono, dtype is arbitrary);
      * labels -- a 2-D numpy array of zeros and ones, where each row
        indicates the sound events active in the corresponding waveform.
    The number of rows in the two variables must match.
    The audio is concatenated and compressed in the FLAC format.
    The labels are appended to the FLAC audio file.
    This function relies on ffmpeg.
    """

    # Validate input
    if len(wav) != len(labels):
        raise ValueError("The number of rows in 'wav' and 'labels' must match.")

    # Convert wav to int16, ensuring the correct range
    nClips, nSamples = wav.shape
    if numpy.abs(wav).max() <= 1: wav *= 32768
    wav = numpy.maximum(numpy.minimum(wav, 32767), -32768).astype("int16")

    # Convert labels to bit arrays
    nLabels = labels.shape[1]
    labels = labels.astype("uint8")
    nBytes = (nLabels - 1) / 8 + 1
    bytes = numpy.zeros((nClips, nBytes), dtype = "uint8")
    for i in xrange(nLabels):
        bytes[:, i / 8] += labels[:,i] << (i % 8)

    # Write file
    wavfile.write(filename + ".wav", 16000, wav.ravel())
    subprocess.check_output("ffmpeg -i %s.wav -c:a flac -y %s && rm %s.wav" % (filename, filename, filename), shell = True)
    with smart_open(filename, "ab") as f:
        f.write(struct.pack("<%dB" % bytes.size, *bytes.ravel()))
        f.write(struct.pack("<3i", nClips, nSamples, nLabels))
