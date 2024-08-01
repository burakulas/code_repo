# Files in the repository: #

. *0_det_synth_lc.sh*: Detached eclipsing binary light curve construction (Sec. 2.1.1).

. *1_sdet_synth_lc.sh*: Semidetached eclipsing binary light curve construction (Sec 2.1.1).

. *2_artificial_pulse.py*: Noise addition, artificial pulsation, and annotation file creation (Sec 2.1.2). 

. *3_random_shift.py*: Random shifting of images and corresponding XML annotations (Sec 2.1.2). 

. *4_npcnn.py*: Non-pretrained CNN-based object detection model (Sec. 3.5).

. *5_detocs_k.py*: Detection on short cadence Kepler data (Sec. 4).

### Detection on *Kepler* Eclipsing Binaries ###

 [*kepler_ebs_sample.csv*](https://github.com/burakulas/detocs/blob/main/assets/kepler_ebs_sample.csv) is a sample that the code can read as input. It creates a folder with the timestamp in the name and moves the images with annotations there. A summary file per target is also created to check the confidence values. Run by `python detocs_k.py`
