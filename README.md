# Files in the repository: #

1. *det_synth_lc.sh*: Detached eclipsing binary light curve construction (Sec. 3.2.1).

2. *sdet_synth_lc.sh*: Semidetached eclipsing binary light curve construction (Sec 3.2.1).

3. *artificial_pulse.py*: Noise addition, artificial pulsation, and annotation file creation (Sec 3.2.2). 

4. *random_shift.py*: Random shifting of images and corresponding XML annotations (Sec 3.2.2). 

5. *npcnn.py*: Non-pretrained CNN-based object detection model (Sec. 4.5).

6. *detocs_k.py*: Detection on short cadence Kepler data. A working sample of the input file is given in [*assets/kepler_ebs_sample.csv*](https://github.com/burakulas/code_repo/blob/main/assets/kepler_ebs_sample.csv) (Sec. 5).



### Citing ###

If you use the above codes in your work or research, please cite them using the following BibTeX entry.

```bibtex
@ARTICLE{2025A&A...695A..81U,
       author = {{Ula{\c{s}}}, B. and {Szklen{\'a}r}, T. and {Szab{\'o}}, R.},
        title = "{Detection of oscillation-like patterns in eclipsing binary light curves using neural network-based object detection algorithms}",
      journal = {\aap},
     keywords = {methods: data analysis, techniques: image processing, binaries: eclipsing, stars: oscillations, Astrophysics - Solar and Stellar Astrophysics},
         year = 2025,
        month = mar,
       volume = {695},
          eid = {A81},
        pages = {A81},
          doi = {10.1051/0004-6361/202452020},
archivePrefix = {arXiv},
       eprint = {2501.17538},
 primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025A&A...695A..81U},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
