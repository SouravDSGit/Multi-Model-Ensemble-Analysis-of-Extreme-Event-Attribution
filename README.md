# Multi-Model Ensemble Analysis for Extreme Event Attribution

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SouravDSGit/Multi-Model-Ensemble-Analysis-of-Extreme-Event-Attribution.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An educational Google Colab notebook** demonstrating how to evaluate multiple climate models, identify systematic biases, and use machine learning to create improved ensemble predictions for extreme events.

## 📚 What You'll Learn

This notebook teaches you how to:

- ✅ Compare multiple climate models (CMIP6-style) against reanalysis data
- ✅ Identify systematic biases in temperature and precipitation extremes
- ✅ Use AI to detect extreme events across different models
- ✅ Apply Random Forest for bias correction (40% error reduction!)
- ✅ Rank models by their skill at predicting extreme events
- ✅ Create optimized multi-model ensembles using machine learning

**Perfect for:** Climate model developers, Earth system scientists learning ensemble methods, researchers evaluating model performance, students studying climate model intercomparison.

## 🌍 Background: Why Multi-Model Ensembles Matter

### The Problem with Single Models

No single climate model is perfect. Each model has:
- **Different physics**: How clouds, radiation, convection are represented
- **Different biases**: Systematic over/underestimation of variables
- **Different strengths**: Model A might be good at temperature, Model B at precipitation

### The Power of Ensembles

Instead of trusting one model, we use **multiple models** together:
- 📊 **Reduces uncertainty**: Average out individual model errors
- 🎯 **Improves predictions**: Ensemble mean often more accurate than any single model
- 🔍 **Reveals biases**: See where models systematically disagree
- 💡 **Informs development**: Understand what physics needs improvement

**Real-world application:** IPCC climate assessments use multi-model ensembles from CMIP (Coupled Model Intercomparison Project) with 30+ models worldwide.

### What This Notebook Does

We evaluate **4 major climate models** (simulating CMIP6):
1. **CESM2** (NCAR, USA)
2. **UKESM1** (UK Met Office)
3. **MPI-ESM** (Max Planck Institute, Germany)
4. **GFDL-CM4** (NOAA, USA)

Then we:
- Compare them against **ERA5 reanalysis** (the "truth")
- Identify their **systematic biases**
- Use **machine learning** to create a better ensemble
- Focus on **extreme events** (heatwaves, heavy precipitation)

## 🚀 Getting Started

### Option 1: Google Colab (Recommended - No Setup!)

1. Click the "Open in Colab" badge at the top
2. Click **Runtime → Run All**
3. Wait ~8-12 minutes for execution
4. Explore the results!

**Zero installation required!** Everything runs in your browser.

### Option 2: Local Jupyter Notebook

```bash
# Clone repository
git clone https://github.com/SouravDSGit/Multi-Model-Ensemble-Analysis-of-Extreme-Event-Attribution.git
cd Multi-Model-Ensemble-Analysis-of-Extreme-Event-Attribution

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook Multi-Model-Ensemble-Analysis-of-Extreme-Event-Attribution.ipynb
```

**Requirements:**
- Python 3.8+
- TensorFlow 2.x
- scikit-learn 1.0+
- Standard scientific Python stack

## 📖 What This Notebook Does

### Complete Workflow (12 Sections)

#### **Section 1-2: Setup & Import**
- Installs all required packages automatically
- Imports TensorFlow, scikit-learn, and scientific Python libraries
- Sets up the environment for climate data analysis

#### **Section 3: Multi-Model Data Generation**
- Creates 35 years of synthetic climate data (1980-2014)
- Simulates 4 different climate models with realistic biases
- Generates ERA5 "truth" as reference
- Includes heatwaves and extreme precipitation events

**Models simulated:**
- **CESM2**: Warm bias (+1.5°C), overestimates heatwaves
- **UKESM1**: Cool bias (-0.5°C), underestimates precip extremes
- **MPI-ESM**: Cool bias (-1.0°C), wet bias
- **GFDL-CM4**: Minimal bias, best overall performance

**Why synthetic data?**
- Real CMIP6 data requires large downloads (100+ GB)
- Complex data processing and format conversion
- This synthetic data has same statistical properties as real CMIP6
- Perfect for learning and demonstration

#### **Section 4: Model Bias Analysis**
- Calculates mean biases for each model
- Computes RMSE and correlation with ERA5
- Quantifies extreme event frequency biases
- Creates comprehensive bias summary tables

**You'll see:**
- Which models run warm or cool
- Which models are too wet or dry
- How extreme event frequencies differ
- Model skill rankings

#### **Section 5: Multi-Model Comparison Visualization**
- Creates publication-quality comparison plots
- Shows temperature and precipitation time series
- Displays bias bar charts for all models
- Visualizes model skill metrics

**Output:** `multimodel_comparison.png` - comprehensive 9-panel figure

#### **Section 6: AI Extreme Event Detector**
- Builds neural network for automatic extreme detection
- Simulates ClimateNet-style approach
- Trains on data from all 4 models
- Achieves 0.85+ AUC for extreme detection

**Why AI detection?**
- Consistent extreme identification across models
- Learns complex patterns (non-linear thresholds)
- Can detect multiple extreme types simultaneously

#### **Section 7: Machine Learning Bias Correction**
- Engineers features from multi-model ensemble
- Trains Random Forest for bias correction
- Learns which models to trust in different situations
- Creates optimized ensemble predictions

**Key insight:** ML learns that ensemble spread (how much models disagree) predicts uncertainty!

#### **Section 8: Feature Importance Analysis**
- Reveals which models contribute most to accuracy
- Shows that ensemble statistics matter (mean, spread)
- Demonstrates seasonal dependencies
- Interprets what the ML learned

**Physical interpretation:** Models with better variability get higher weight—makes sense!

#### **Section 9: Bias Correction Results**
- Compares uncorrected vs ML-corrected ensemble
- Shows 30-40% error reduction
- Visualizes improvement in predictions
- Creates before/after comparison plots

**Output:** `bias_correction_results.png` - detailed performance analysis

#### **Section 10: Model Ranking for Extremes**
- Calculates extreme event skill scores
- Uses meteorological metrics (POD, FAR, CSI)
- Ranks models by their extreme prediction skill
- Determines best models for different extreme types

**Metrics explained:**
- **POD** (Probability of Detection): Did model catch the extreme?
- **FAR** (False Alarm Ratio): How many false alarms?
- **CSI** (Critical Success Index): Overall skill score

#### **Section 11: Model Ranking Visualization**
- Creates skill ranking bar charts
- Shows POD vs FAR tradeoffs
- Compares heatwave vs precipitation skill
- Visualizes overall performance

**Output:** `model_extreme_ranking.png` - model comparison

#### **Section 12: Summary Report**
- Generates comprehensive text summary
- Lists all findings and metrics
- Provides context for applications
- Explains implications for model development

**Output:** `multimodel_summary.txt` - complete analysis report

### 📊 What Results You'll Get

After running the notebook:

**3 Publication-Quality Figures:**
1. `multimodel_comparison.png` - Model biases and comparison
2. `bias_correction_results.png` - ML enhancement results
3. `model_extreme_ranking.png` - Model skill rankings

**1 Summary Report:**
- `multimodel_summary.txt` - Complete analysis writeup

**Typical Performance:**
- **Temperature Bias Correction**: 30-40% MAE reduction
- **Precipitation Bias Correction**: 30-35% MAE reduction
- **Extreme Event Detection**: 0.85+ AUC
- **Best Model Identified**: GFDL-CM4 (minimal biases)
- **Ensemble Improvement**: Corrected ensemble outperforms all individual models

## 🔬 Technical Deep Dive

### Why These 4 Models?

We simulate models representing different modeling centers:

**CESM2 (Community Earth System Model)**
- Developed by NCAR (National Center for Atmospheric Research)
- High resolution, good for regional climate
- Known warm bias in some configurations

**UKESM1 (UK Earth System Model)**
- Developed by UK Met Office
- Advanced Earth system components (carbon cycle, chemistry)
- Good spatial patterns

**MPI-ESM (Max Planck Institute Earth System Model)**
- Developed in Germany
- Excellent ocean-atmosphere coupling
- Conservative temperature predictions

**GFDL-CM4 (Geophysical Fluid Dynamics Laboratory)**
- Developed by NOAA
- Strong atmospheric physics
- Often closest to observations

### Multi-Model Statistics

**Ensemble Mean:**
- Simple average of all models
- Often better than any single model
- Reduces random errors

**Ensemble Spread:**
- How much models disagree
- Large spread = high uncertainty
- Useful for probability forecasts

**Ensemble Weighting:**
- Give more weight to better models
- Can use performance-based weights
- ML learns optimal weights automatically

### Machine Learning Approach

**Random Forest for Bias Correction:**

**Why Random Forest?**
- Handles non-linear relationships
- Robust to outliers
- Provides feature importance
- No complex hyperparameter tuning needed

**What it learns:**
- Which models to trust for different conditions
- How to combine ensemble statistics
- Seasonal dependencies in biases
- When ensemble spread indicates uncertainty

**Input Features:**
- Individual model predictions (4 models)
- Ensemble mean
- Ensemble spread (standard deviation)
- Ensemble max/min
- Day of year (seasonal cycle)
- Month

**Output:**
- Bias-corrected prediction (closer to ERA5 truth)

### Extreme Event Metrics

**Classification Metrics:**

```
Probability of Detection (POD) = Hits / (Hits + Misses)
False Alarm Ratio (FAR) = False Alarms / (Hits + False Alarms)
Critical Success Index (CSI) = Hits / (Hits + Misses + False Alarms)
```

**What they mean:**
- **High POD, Low FAR** = Ideal (catches extremes, few false alarms)
- **High POD, High FAR** = Over-predicts (catches extremes but many false alarms)
- **Low POD, Low FAR** = Under-predicts (misses extremes, but when it predicts, it's right)

## 🎓 Educational Value

### For Climate Scientists
- Learn multi-model evaluation techniques
- Understand ensemble methods and statistics
- See how to quantify model biases systematically
- Apply ML for bias correction

### For Data Scientists
- Work with climate model output
- Learn domain-specific evaluation metrics
- See ensemble learning in a physics context
- Understand feature engineering for Earth science

### For Model Developers
- Identify where your model needs improvement
- Compare against other models objectively
- Use bias correction as interim solution
- Guide physics parameterization development

### For Students
- Complete climate model intercomparison project
- Learn both traditional and ML methods
- Understand CMIP framework
- Applicable to thesis/coursework

## 💡 Extension Ideas

Want to extend this project?

1. **Use Real CMIP6 Data**: Download actual model output from ESGF
2. **More Models**: Expand from 4 to 10+ models
3. **Spatial Analysis**: Move from point to gridded data
4. **More Variables**: Add wind, humidity, sea ice, etc.
5. **Process-Level Metrics**: Evaluate clouds, convection, radiation
6. **Climate Change Scenarios**: Compare historical vs future projections
7. **Regional Focus**: Zoom in on specific regions (Arctic, tropics, etc.)
8. **Uncertainty Quantification**: Add confidence intervals to ensemble
9. **Deep Learning**: Try neural networks instead of Random Forest
10. **Real-Time Evaluation**: Set up automated assessment pipeline

## 🤝 Contributing

Educational contributions welcome!

- 🐛 Bug reports → Open an issue
- 💡 Improvements → Submit a pull request
- 📚 Documentation → Always appreciated
- 🎓 Teaching examples → Share your use case

## 📄 License

MIT License - free for education, research, and commercial use!

## 🙏 Acknowledgments & References

This notebook was created as an educational resource based on:

### Climate Model Intercomparison
- **CMIP6**: Eyring et al. (2016), "Overview of the Coupled Model Intercomparison Project Phase 6", *Geoscientific Model Development*
- **Model Evaluation**: Flato et al. (2013), "Evaluation of Climate Models" in IPCC AR5
- **Ensemble Methods**: Knutti et al. (2010), "Challenges in combining projections from multiple climate models", *Journal of Climate*

### Machine Learning for Climate
- **ML Bias Correction**: Ahn et al. (2021), "A deep learning model for the prediction of rainfall", *Remote Sensing*
- **Random Forests**: Breiman (2001), "Random Forests", *Machine Learning*
- **Feature Importance**: Lundberg & Lee (2017), "A unified approach to interpreting model predictions", *NIPS*

### Extreme Event Analysis
- **Extreme Detection**: Prabhat et al. (2015), "ClimateNet: An expert-labeled dataset for extreme weather event detection", *Scientific Data*
- **Verification Metrics**: Wilks (2011), "Statistical Methods in the Atmospheric Sciences"
- **Heatwave Definition**: Perkins & Alexander (2013), "On the measurement of heat waves", *Journal of Climate*

### Data Sources (for real applications)
- **ERA5 Reanalysis**: Hersbach et al. (2020), "The ERA5 global reanalysis", *QJRMS*
- **CMIP6 Archive**: Earth System Grid Federation (ESGF) at https://esgf-node.llnl.gov
- **Model Documentation**: 
  - CESM2: Danabasoglu et al. (2020)
  - UKESM1: Sellar et al. (2019)
  - MPI-ESM: Mauritsen et al. (2019)
  - GFDL-CM4: Held et al. (2019)

### Software & Tools
- **TensorFlow**: Abadi et al. (2015), "TensorFlow: Large-scale machine learning"
- **scikit-learn**: Pedregosa et al. (2011), "Scikit-learn: Machine learning in Python", *JMLR*
- **NumPy**: Harris et al. (2020), "Array programming with NumPy", *Nature*
- **Pandas**: McKinney (2010), "Data structures for statistical computing in Python"
- **Matplotlib**: Hunter (2007), "Matplotlib: A 2D graphics environment"

### Inspiration
- CMIP Analysis Tools (PCMDI)
- ESMValTool (Earth System Model Evaluation Tool)
- Climate model evaluation workshops
- IPCC Working Group I methodology

## 📧 Contact

- **Issues**: GitHub Issues for bugs/questions
- **Discussions**: GitHub Discussions for ideas
- **Email**: [soumukhcivil@gmail.com]

## 🌟 Citation

If you use this notebook for research or teaching:

```bibtex
@software{multimodel_ensemble,
  author = {Sourav Mukherjee},
  title = {Multi-Model Ensemble Analysis for Extreme Event Attribution},
  year = {2025},
  url = {https://github.com/SouravDSGit/Multi-Model-Ensemble-Analysis-of-Extreme-Event-Attribution}
}
```

---

**Happy Learning! 🌍**

*Note: This project uses synthetic data for educational purposes. For research publications or operational forecasting, use actual CMIP6 data from ESGF and real reanalysis products (ERA5, MERRA-2, JRA-55).*

---

## 🔗 Related Projects

- [Project 1: AI Emulator for Atmospheric Blocking](https://github.com/SouravDSGit/AI-Emulator-for-Atmospheric-Blocking-Events/tree/main)
- [Project 3: Graph Neural Networks for Teleconnections](../gnn-teleconnections/)

**Together, these three projects demonstrate:**
- Computational efficiency (Project 1)
- Systematic evaluation (Project 2)
- Spatial AI (Project 3)

All three applicable to Earth system model development and extreme event prediction!
