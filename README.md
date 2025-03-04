# Files in the repository: #

1. *det_synth_lc.sh*: Detached eclipsing binary light curve construction (Sec. 3.2.1).

2. *sdet_synth_lc.sh*: Semidetached eclipsing binary light curve construction (Sec 3.2.1).

3. *artificial_pulse.py*: Noise addition, artificial pulsation, and annotation file creation (Sec 3.2.2). 

4. *random_shift.py*: Random shifting of images and corresponding XML annotations (Sec 3.2.2). 

5. *npcnn.py*: Non-pretrained CNN-based object detection model (Sec. 4.5).

6. *detocs_k.py*: Detection on short cadence Kepler data. A working sample of the input file is given in [*assets/kepler_ebs_sample.csv*](https://github.com/burakulas/code_repo/blob/main/assets/kepler_ebs_sample.csv) (Sec. 5).


### Citing ###

If you use the above codes in your work or research, please use the following BibTeX entry.

```bibtex
@ARTICLE{2025arXiv250117538U,
       author = {{Ula{\c{s}}}, Burak and {Szklen{\'a}r}, Tam{\'a}s and {Szab{\'o}}, R{\'o}bert},
        title = "{Detection of Oscillation-like Patterns in Eclipsing Binary Light Curves using Neural Network-based Object Detection Algorithms}",
         year = 2025,
        month = jan,
          doi = {https://doi.org/10.1051/0004-6361/202452020},
          eid = {arXiv:2501.17538},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv250117538U},
}
```
