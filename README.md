# TC_new
# Research On Speaker Timber Conversion Method Based On PPG 

## Model Architecture

The main significance of this work is that we could generate a target speaker's utterances without parallel data like <source's wav, target's wav>, <wav, text> or <wav, phone>, but only waveforms of the target speaker.
(To make these parallel datasets needs a lot of effort.)
All we need in this project is a number of waveforms of the target speaker's utterances and only a small set of <wav, phone> pairs from a number of anonymous speakers.

The model architecture consists of two modules:

1. Net1(phoneme classification) classify someone's utterances to one of phoneme classes at every timestep.
   * Phonemes are speaker-independent while waveforms are speaker-dependent.
2. Net2(speech synthesis) synthesize speeches of the target speaker from the phones.

### Net1 is a classifier.

* Process: wav -> spectrogram -> mfccs -> phoneme dist.
* Net1 classifies spectrogram to phonemes that consists of 60 English phonemes at every timestep.
  * For each timestep, the input is log magnitude spectrogram and the target is phoneme dist.
* Objective function is cross entropy loss.
* MITIMT used.
  * contains 630 speakers' utterances and corresponding phones that speaks similar sentences.
* Over 70% test accuracy

### Net2 is a synthesizer.

Net2 contains Net1 as a sub-network.

* Process: net1(wav -> spectrogram -> mfccs -> phoneme dist.) -> spectrogram -> wav
* Net2 synthesizes the target speaker's speeches.
  * The input/target is a set of target speaker's utterances.
* Since Net1 is already trained in previous step, the remaining part only should be trained in this step.
* Loss is reconstruction error between input and target. (L2 distance)
* Datasets
  * Target(anonymous female): Arctic(public)
* Griffin-Lim reconstruction when reverting wav from spectrogram.

## Implementations

### Requirements

* python 3.7
* pytorch == 1.5
* librosa == 0.7.2

### Settings

* sample rate: 16,000Hz
* window length: 25ms
* hop length: 5ms

### Procedure

* Train phase: Net1 and Net2 should be trained sequentially.
  * Train1(training Net1)
    * Run `train_net1.py` to train and `test_net1.py` to test.
  * Train2(training Net2)
    * Run `train_net2.py` to train and `test_net2.py` to test.
      * Train2 should be trained after Train1 is done!
* Convert phase: feed forward to Net2
  * Run `convert.py` to get result samples.
  * Check Tensorboard's audio tab to listen the samples.
  * Take a look at phoneme dist. visualization on Tensorboard's image tab.
    * x-axis represents phoneme classes and y-axis represents timesteps
    * the first class of x-axis means silence.
