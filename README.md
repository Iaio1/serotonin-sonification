# **The Sound of Serotonin**

### Background
Fast-Scan Cyclic Voltammetry (FSCV) is a bioelectrochemical technique widely regarded as both technically and conceptually challenging. It allows the measurement of spontaneous serotonin release from various cell types, including skin cells [^1], gut cells [^2], and mice serotonergic neurons [^3], and more recently, from human stem cell–derived serotonergic neurons [^4].

**Sonification** refers to the use of non-speech audio to convey information ([Sonification Handbook](https://sonification.de/handbook/)). By converting data relationships into acoustic signals, sonification can enhance communication and facilitate intuitive understanding of complex data patterns. 

While several efforts have been made to improve the accessibility and outreach of FSCV data, the use of sonification as an interpretive or outreach tool has not yet been explored.

---
### Project Aim
This project aims to transform pre-existing spontaneous FSCV datasets collected by the lab into musical representations, building upon the previously developed [NeuroStemVolt](https://github.com/pablopriet/NeuroStemVolt) software. The goal is to design a **mapping strategy (processing pipeline)** that converts serotonin driven electrical activity into sound or music.

---
### Objectives
The students will:

1. **Organize and preprocess** spontaneous serotonin release time-series data.  
2. **Develop software tools** (in Python) extending the *NeuroStemVolt* framework.  
3. **Design an interactive user interface (UI)** that allows flexible mapping of data parameters to musical variables, such as:  
   - Pitch  
   - Loudness  
   - Timbre  
   - Spatialization  
4. **Generate playable MIDI files** by converting binned FSCV data from various cell types, including:  
   - Skin cells [^1]  
   - Gut cells [^2]  
   - Mouse serotonergic neurons [^3]  
   - Human stem cell–derived serotonergic neurons [^4]  
1. **Integrate the resulting MIDI files** into a Digital Audio Workstation (DAW), e.g. Ableton, Logic, or Reaper, to experiment with instruments, effects, and spatialization.

---
### Expected Outcomes
The resulting software will enable users to transform bioelectric serotonin data into expressive musical compositions. Ideally, combining different types of cell releasing patterns into a single composition.
The MIDI outputs can be used creatively in performance or education settings, serving as both scientific visualization and artistic expression.

---
### Dissemination & Impact
The work produced in this project is expected to lead to:

- A **research publication** detailing the methodology and creative process.  
- A **presentation at** *The Great Exhibition Road Festival*.  
- Potential **outreach applications** for public engagement, particularly in schools, raising awareness about depression, neuroscience, and bioelectrochemistry.

--- 
**Additional Resources**  
- [Sonification Handbook](https://sonification.de/handbook/)  
- [NeuroStemVolt Software Repository](https://github.com/pablopriet/NeuroStemVolt)

[^1]: **Skin cells** – T.A. et al., *British Journal of Dermatology*, 190(6):e88, 2024. [https://academic.oup.com/bjd/article/190/6/e88/7675825](https://academic.oup.com/bjd/article/190/6/e88/7675825)

[^2]: **Gut cells** – B.B. et al., *Analytical Chemistry*, 97(8):12345–12356, 2024. [https://pubs.acs.org/doi/full/10.1021/acs.analchem.4c06033](https://pubs.acs.org/doi/full/10.1021/acs.analchem.4c06033)

[^3]: **Mouse serotonergic neurons** – P.H. et al., *Analytical Chemistry*, 81(17):7046–7055, 2009. [https://pubs.acs.org/doi/abs/10.1021/ac9018846](https://pubs.acs.org/doi/abs/10.1021/ac9018846)

[^4]: **Human stem cell–derived serotonergic neurons** – B. B et al., SSRN preprint, 2025. [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5390878](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5390878)


---

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-xyz`).
3. Commit your changes.
4. Push to your fork and submit a pull request.

---

## Dependencies

Minimal environment (see `environment.yml`):

- python >= 3.11
- numpy
- pandas
- matplotlib
- pyqt
- scipy
- pip (for packaging)
- pyinstaller (for building executables)

---

## License

[MIT License](LICENSE)  

---

## Acknowledgements

Developed by...
For questions or support, contact ...

### References



